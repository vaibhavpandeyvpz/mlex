# `mlex-node`

Internal NAPI-RS glue crate that compiles [`mlex`](../mlex) into a native Node.js addon (`.node` binary). It is **not published to crates.io** — see below for why — and is not meant to be depended on directly by other Rust crates.

## Why this isn't published

Unlike `mlex`, this crate has no useful Rust API surface:

- It's built as a `cdylib` (`crate-type = ["cdylib"]`), which other Rust crates can't link against the normal way — `cdylib`-only crates produce a dynamic library, not an `rlib`, so there's nothing for `cargo` to consume even if you added it as a dependency.
- Every `#[napi]`-annotated type/function exists purely to describe a JS-facing shape (e.g. `JsChatMessage`, `MlexModel`); none of it is meaningful outside of a Node.js runtime calling into the compiled addon.
- Its only real "distribution" mechanism is the compiled `.node` binary, which ships inside the [`mlex` npm package](../../packages/node) — not as Rust source.

Publishing it to crates.io would add a crate with a misleading appearance of being a usable library, so `Cargo.toml` sets `publish = false` and it stays workspace-internal.

## What actually gets published

- **Rust:** [`mlex`](../mlex) → crates.io.
- **Node.js:** the compiled `.node` addon from this crate, plus the hand-written TypeScript types/wrapper in [`packages/node`](../../packages/node) → npm, as the [`mlex`](https://www.npmjs.com/package/mlex) package.

## Building locally

```bash
cd packages/node
npm run build   # napi build --platform --release --manifest-path ../../crates/mlex-node/Cargo.toml ...
```

This compiles `mlex-node` in release mode and drops the resulting `.node` binary plus regenerated `index.js`/`index.d.ts` into `packages/node/`, ready for `npm test` or `npm publish`.

See the [`mlex` npm package README](../../packages/node/README.md) for the actual public JS/TS API, and the [top-level project README](../../README.md) for the full picture.
