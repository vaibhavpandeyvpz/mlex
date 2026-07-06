//! Stateless, server-style prompt-cache pool.
//!
//! Mirrors how the hosted chat APIs actually expose prompt caching: the
//! caller sends the *full* message list on every call (no session handle),
//! and the server transparently keeps a small pool of recently-used
//! prefixes' KV state around so that two calls sharing a prefix (most
//! commonly: the same conversation's next turn, but also just two
//! unrelated calls sharing a system prompt) skip re-running the model over
//! the shared part. OpenAI does this fully automatically; Anthropic adds
//! explicit `cache_control` breakpoint hints - this pool supports the
//! automatic (unpinned, LRU + TTL) case, which is enough to reproduce
//! `Conversation`'s old speedup without requiring a caller-held handle.
//!
//! [`LayerCache`] cloning is a cheap refcount bump per `Array` field (MLX
//! arrays are `shared_ptr`-backed - see `Array::clone` in `array.rs`), so
//! forking a pooled entry's caches into a live generation call, or storing
//! a fresh snapshot back into the pool, costs O(num_layers) rather than
//! O(cache size). That's what makes a multi-entry pool practical here.

use std::time::{Duration, Instant};

use crate::models::cache::LayerCache;

/// One cached prefix: the token ids it represents, the per-layer KV/state
/// caches after processing them, and how much of the multimodal media
/// queue (in placeholder order) has already been fed through the towers
/// and spliced into those caches.
#[derive(Clone)]
pub struct CacheEntry {
    pub ids: Vec<u32>,
    pub caches: Vec<LayerCache>,
    pub fed_images: usize,
    pub fed_audios: usize,
    last_used: Instant,
    /// Exempt from LRU/TTL eviction (mirrors Anthropic's explicit
    /// `cache_control: {type: "ephemeral"}` breakpoints). Not yet wired to
    /// a public cache-hint API - see `ChatMessage`/`generate_cached` TODO.
    pub pinned: bool,
}

/// Pool sizing knobs, split out from [`PromptCachePool`] so callers (e.g.
/// the Node binding) can override [`PromptCacheConfig::default`] piecemeal
/// without reaching into `Duration`/pool internals.
#[derive(Clone, Copy, Debug)]
pub struct PromptCacheConfig {
    pub max_entries: usize,
    pub ttl: Duration,
    pub min_cacheable_tokens: usize,
}

impl Default for PromptCacheConfig {
    /// Mirrors [`PromptCachePool::with_defaults`]: 16 entries, 5 minute
    /// idle TTL, 8-token minimum.
    fn default() -> Self {
        PromptCacheConfig {
            max_entries: 16,
            ttl: Duration::from_secs(5 * 60),
            min_cacheable_tokens: 8,
        }
    }
}

/// A small, in-process pool of cached prompt prefixes, keyed by
/// longest-common-prefix match on token ids rather than a fixed-block
/// content hash (appropriate at this scale - tens of entries, not a
/// fleet-wide server cache).
pub struct PromptCachePool {
    entries: Vec<CacheEntry>,
    max_entries: usize,
    ttl: Duration,
    min_cacheable_tokens: usize,
}

impl PromptCachePool {
    pub fn new(max_entries: usize, ttl: Duration, min_cacheable_tokens: usize) -> Self {
        PromptCachePool {
            entries: Vec::new(),
            max_entries,
            ttl,
            min_cacheable_tokens,
        }
    }

    /// Default pool sizing: 16 entries, 5 minute idle TTL (mirrors
    /// Anthropic's ephemeral cache-control default lifetime), and an
    /// 8-token minimum before a prefix is worth keeping around at all -
    /// trivially short prompts (a couple of words) cost nothing to
    /// recompute, so caching them just churns pool slots that a
    /// meaningfully-sized prefix could otherwise occupy. This is a much
    /// lower bar than OpenAI's real 1024-token cutoff, which exists for a
    /// different reason (their cache is a shared, metered, multi-tenant
    /// resource; this one is just process-local memory).
    pub fn with_defaults() -> Self {
        Self::from_config(PromptCacheConfig::default())
    }

    pub fn from_config(config: PromptCacheConfig) -> Self {
        Self::new(config.max_entries, config.ttl, config.min_cacheable_tokens)
    }

    /// Find the entry whose `ids` is the longest exact prefix of `ids`,
    /// clone it (cheap - see module docs), and return it alongside how
    /// many leading tokens of `ids` it covers (`entry.ids.len()`).
    ///
    /// Note this deliberately requires an exact-prefix match, not merely
    /// the longest *common* prefix: a KV cache is append-only, so an entry
    /// whose `ids` diverges from the query partway through (rather than
    /// being a strict prefix of it) holds state for tokens that don't
    /// belong in the new sequence at all and can't be reused - only a
    /// clean reset (equivalent to a pool miss) is correct there.
    ///
    /// Returns `None` (equivalent to a fresh/cold start) if no entry is a
    /// prefix of `ids`, or the pool is empty.
    pub fn find_longest_prefix(&mut self, ids: &[u32]) -> Option<(CacheEntry, usize)> {
        self.evict_expired();

        let mut best: Option<usize> = None; // entry index
        for (i, entry) in self.entries.iter().enumerate() {
            if !entry.ids.is_empty()
                && is_prefix(&entry.ids, ids)
                && best
                    .map(|b| entry.ids.len() > self.entries[b].ids.len())
                    .unwrap_or(true)
            {
                best = Some(i);
            }
        }

        let idx = best?;
        self.entries[idx].last_used = Instant::now();
        let shared = self.entries[idx].ids.len();
        Some((self.entries[idx].clone(), shared))
    }

    /// Insert or refresh a pool entry for `ids`. If an existing entry's
    /// `ids` is a prefix of (or equal to) the new `ids` - the common
    /// "extend this lineage by one more turn" case - it's replaced in
    /// place rather than growing the pool unboundedly across a long
    /// conversation.
    ///
    /// A no-op (besides refreshing an existing entry - see below) if
    /// `ids` is shorter than [`Self::min_cacheable_tokens`]: below that,
    /// there's nothing worth keeping a cache slot warm for. Note this
    /// only gates *new* lineages - an existing entry that already cleared
    /// the bar is still refreshed/extended even if, hypothetically,
    /// `ids` were to shrink (it never does in practice: `ids` is always
    /// the previous call's ids plus newly-generated tokens).
    pub fn insert_or_update(
        &mut self,
        ids: Vec<u32>,
        caches: Vec<LayerCache>,
        fed_images: usize,
        fed_audios: usize,
        pinned: bool,
    ) {
        let now = Instant::now();
        if ids.len() < self.min_cacheable_tokens
            && !self.entries.iter().any(|e| is_prefix(&e.ids, &ids))
        {
            return;
        }
        if let Some(existing) = self.entries.iter_mut().find(|e| is_prefix(&e.ids, &ids)) {
            existing.ids = ids;
            existing.caches = caches;
            existing.fed_images = fed_images;
            existing.fed_audios = fed_audios;
            existing.last_used = now;
            existing.pinned = existing.pinned || pinned;
            return;
        }

        self.entries.push(CacheEntry {
            ids,
            caches,
            fed_images,
            fed_audios,
            last_used: now,
            pinned,
        });
        self.evict_lru_if_over_capacity();
    }

    fn evict_expired(&mut self) {
        let ttl = self.ttl;
        let now = Instant::now();
        self.entries
            .retain(|e| e.pinned || now.duration_since(e.last_used) < ttl);
    }

    fn evict_lru_if_over_capacity(&mut self) {
        while self.entries.len() > self.max_entries {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.pinned)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    self.entries.remove(i);
                }
                // Every entry is pinned - nothing evictable, stop trying.
                None => break,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn is_prefix(shorter: &[u32], longer: &[u32]) -> bool {
    shorter.len() <= longer.len() && shorter == &longer[..shorter.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_caches() -> Vec<LayerCache> {
        Vec::new()
    }

    #[test]
    fn find_longest_prefix_picks_the_best_match() {
        let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        pool.insert_or_update(vec![1, 2, 3, 4, 5], empty_caches(), 0, 0, false);

        // Both entries share a prefix with [1,2,3,4,5,6]; the longer one
        // must win.
        let (entry, shared) = pool.find_longest_prefix(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(entry.ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(shared, 5);
    }

    #[test]
    fn find_longest_prefix_returns_none_when_no_overlap() {
        let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        assert!(pool.find_longest_prefix(&[9, 9, 9]).is_none());
    }

    #[test]
    fn insert_or_update_extends_existing_lineage_in_place() {
        let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 0);
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        pool.insert_or_update(vec![1, 2, 3, 4, 5], empty_caches(), 1, 0, false);
        assert_eq!(
            pool.len(),
            1,
            "extending a lineage should not grow the pool"
        );
    }

    #[test]
    fn insert_or_update_evicts_lru_over_capacity() {
        let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
        pool.insert_or_update(vec![1], empty_caches(), 0, 0, false);
        pool.insert_or_update(vec![2], empty_caches(), 0, 0, false);
        // Touch entry [1] so [2] becomes the least-recently-used.
        pool.find_longest_prefix(&[1]);
        pool.insert_or_update(vec![3], empty_caches(), 0, 0, false);

        assert_eq!(pool.len(), 2);
        assert!(
            pool.find_longest_prefix(&[2]).is_none(),
            "LRU entry should have been evicted"
        );
        assert!(pool.find_longest_prefix(&[1]).is_some());
        assert!(pool.find_longest_prefix(&[3]).is_some());
    }

    #[test]
    fn pinned_entries_survive_lru_pressure() {
        // Capacity 2: [1] (pinned) + [2], then adding [3] must evict [2]
        // (the only unpinned entry) rather than the pinned [1], even
        // though [1] is now the least-recently-touched by wall-clock time.
        let mut pool = PromptCachePool::new(2, Duration::from_secs(300), 0);
        pool.insert_or_update(vec![1], empty_caches(), 0, 0, true);
        pool.insert_or_update(vec![2], empty_caches(), 0, 0, false);
        pool.insert_or_update(vec![3], empty_caches(), 0, 0, false);

        assert_eq!(pool.len(), 2);
        assert!(
            pool.find_longest_prefix(&[1]).is_some(),
            "pinned entry must survive eviction"
        );
        assert!(
            pool.find_longest_prefix(&[2]).is_none(),
            "unpinned entry should have been evicted"
        );
        assert!(pool.find_longest_prefix(&[3]).is_some());
    }

    #[test]
    fn entries_shorter_than_the_minimum_are_not_cached() {
        let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 8);
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        assert!(
            pool.is_empty(),
            "a 3-token entry should be rejected by an 8-token minimum"
        );
        assert!(pool.find_longest_prefix(&[1, 2, 3]).is_none());
    }

    #[test]
    fn a_lineage_becomes_cacheable_once_it_crosses_the_minimum() {
        let mut pool = PromptCachePool::new(16, Duration::from_secs(300), 8);
        // Turn one: 3 tokens, below the minimum - not stored.
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        assert!(pool.is_empty());

        // Turn two extends the same lineage past the minimum - now stored,
        // and reachable by a later exact-prefix lookup.
        let turn_two: Vec<u32> = (1..=10).collect();
        pool.insert_or_update(turn_two.clone(), empty_caches(), 0, 0, false);
        assert_eq!(pool.len(), 1);
        assert!(pool.find_longest_prefix(&turn_two).is_some());
    }

    #[test]
    fn expired_unpinned_entries_are_evicted() {
        let mut pool = PromptCachePool::new(16, Duration::from_millis(1), 0);
        pool.insert_or_update(vec![1, 2, 3], empty_caches(), 0, 0, false);
        std::thread::sleep(Duration::from_millis(5));
        assert!(pool.find_longest_prefix(&[1, 2, 3]).is_none());
    }
}
