// Shared model-discovery + size-gating helper for the Vitest suites,
// mirroring `crates/mlex/tests/common/mod.rs`. Rather than
// re-measuring peak memory independently, this reads the same
// `target/mlex-test-cache/model-memory.json` cache the Rust suite writes
// (which runs first in CI) - if that cache doesn't exist yet (Rust tests
// haven't run), models are conservatively treated as unmeasured/excluded
// rather than guessed at.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

export interface ModelInfo {
  dir: string;
  repoId: string;
  family: string;
  weightsBytes: number;
  peakRssBytes: number | null;
  ciSafe: boolean;
}

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");

function modelsDir(): string {
  return process.env.MLEX_MODELS_DIR ?? join(REPO_ROOT, "models");
}

function maxModelGb(): number {
  const v = process.env.MLEX_MAX_MODEL_GB;
  return v ? Number.parseFloat(v) : 5.0;
}

function includeLargeModels(): boolean {
  return process.env.MLEX_INCLUDE_LARGE_MODELS === "1";
}

function memoryCachePath(): string {
  return join(REPO_ROOT, "target", "mlex-test-cache", "model-memory.json");
}

function loadMemoryCache(): Record<string, number> {
  const path = memoryCachePath();
  if (!existsSync(path)) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8"));
    return parsed.entries ?? {};
  } catch {
    return {};
  }
}

function safetensorsBytes(dir: string): number {
  let total = 0;
  for (const entry of readdirSync(dir)) {
    if (entry.endsWith(".safetensors")) {
      total += statSync(join(dir, entry)).size;
    }
  }
  return total;
}

function modelTypeOf(configPath: string): string {
  try {
    const json = JSON.parse(readFileSync(configPath, "utf-8"));
    return typeof json.model_type === "string" ? json.model_type : "unknown";
  } catch {
    return "unknown";
  }
}

let cachedRegistry: ModelInfo[] | null = null;

/** Discover every locally-downloaded model, classified against the same
 * `MLEX_MAX_MODEL_GB` ceiling and measured-memory cache the Rust suite
 * uses. Returns only `ciSafe` models unless `MLEX_INCLUDE_LARGE_MODELS=1`. */
export function registry(): ModelInfo[] {
  if (cachedRegistry) return cachedRegistry;

  const root = modelsDir();
  const maxBytes = maxModelGb() * 1024 * 1024 * 1024;
  const memCache = loadMemoryCache();
  const out: ModelInfo[] = [];

  if (!existsSync(root)) return [];

  for (const topEntry of readdirSync(root)) {
    if (!topEntry.startsWith("models--")) continue;
    const snapshotsDir = join(root, topEntry, "snapshots");
    if (!existsSync(snapshotsDir)) continue;

    for (const snap of readdirSync(snapshotsDir)) {
      const dir = join(snapshotsDir, snap);
      const configPath = join(dir, "config.json");
      if (!existsSync(configPath)) continue;

      const family = modelTypeOf(configPath);
      const weightsBytes = safetensorsBytes(dir);
      const repoId = topEntry.replace(/^models--/, "").replace("--", "/");

      let ciSafe = false;
      let peakRssBytes: number | null = null;
      if (weightsBytes <= maxBytes * 2) {
        const cacheKey = `${dir}|${weightsBytes}`;
        const measured = memCache[cacheKey];
        if (measured !== undefined) {
          peakRssBytes = measured;
          ciSafe = measured <= maxBytes;
        }
      }

      out.push({ dir, repoId, family, weightsBytes, peakRssBytes, ciSafe });
    }
  }

  cachedRegistry = includeLargeModels() ? out : out.filter((m) => m.ciSafe);
  return cachedRegistry;
}

export function registryForFamily(family: string): ModelInfo[] {
  return registry().filter((m) => m.family === family);
}

// Any one of these tensor-name prefixes present is enough to call a given
// `config.json` sub-dict's capability "real" (not just declared): either
// the classic transformer tower, or the encoder-free "unified" path's
// patch/window embedder - `embed_vision.`/`embed_audio.` alone is the
// simplest and most reliable signal (both encoder shapes always route
// through it), listed alongside the more specific tower/embedder prefixes
// for clarity. Mirrors the Rust loader's own weight-presence gate, so
// checkpoints that declare `vision_config`/`audio_config` for
// architecture-class metadata but were distributed text-only (no matching
// weights at all) aren't mistaken for capable multimodal checkpoints. See
// `crates/mlex/tests/common/mod.rs::has_capability`.
const REQUIRED_WEIGHT_PREFIXES: Record<string, string[]> = {
  vision_config: ["vision_tower.", "vision_embedder.", "embed_vision."],
  audio_config: ["audio_tower.", "embed_audio."],
};

function safetensorsHeaderHasPrefix(path: string, prefix: string): boolean {
  if (!existsSync(path)) return false;
  try {
    const fd = readFileSync(path);
    if (fd.length < 8) return false;
    const headerLen = Number(fd.readBigUInt64LE(0));
    if (fd.length < 8 + headerLen) return false;
    const header = JSON.parse(fd.subarray(8, 8 + headerLen).toString("utf-8"));
    return Object.keys(header)
      .filter((k) => k !== "__metadata__")
      .some((k) => k.startsWith(prefix));
  } catch {
    return false;
  }
}

// Scan every `*.safetensors` shard under `dir` (sharded index, single
// file, and any sidecar shard the index doesn't mention - e.g. OptiQ
// checkpoints' `optiq_vision.safetensors`) for a tensor name starting with
// `prefix`.
function anyTensorHasPrefix(dir: string, prefix: string): boolean {
  const files = new Set<string>();
  const indexPath = join(dir, "model.safetensors.index.json");
  if (existsSync(indexPath)) {
    try {
      const json = JSON.parse(readFileSync(indexPath, "utf-8"));
      const weightMap = json.weight_map ?? {};
      for (const v of Object.values<string>(weightMap)) files.add(v);
    } catch {
      // fall through to directory scan below
    }
  } else {
    files.add("model.safetensors");
  }
  try {
    for (const name of readdirSync(dir)) {
      if (name.endsWith(".safetensors")) files.add(name);
    }
  } catch {
    // ignore, use whatever the index gave us
  }

  return Array.from(files).some((file) =>
    safetensorsHeaderHasPrefix(join(dir, file), prefix),
  );
}

export function hasCapability(dir: string, key: string): boolean {
  try {
    const json = JSON.parse(readFileSync(join(dir, "config.json"), "utf-8"));
    if (json[key] === undefined) return false;
    const prefixes = REQUIRED_WEIGHT_PREFIXES[key];
    return prefixes ? prefixes.some((p) => anyTensorHasPrefix(dir, p)) : true;
  } catch {
    return false;
  }
}
