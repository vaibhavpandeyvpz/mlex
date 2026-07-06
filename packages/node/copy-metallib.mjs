// Post-build step for `npm run build` / `npm run build:debug`.
//
// MLX locates its compiled Metal kernel library (`mlx.metallib`) at runtime
// by checking the absolute path baked in at compile time (the cargo build
// tree — absent on end-user machines) and then falling back to a path
// colocated with the loaded binary. The npm package must therefore ship
// `mlx.metallib` right next to `mlex.js.darwin-arm64.node`, or every load
// on a machine without the build tree fails with
// "Failed to load the default metallib".
//
// `crates/mlex/build.rs` leaves the built metallib under cargo's
// target/<profile>/build/mlex-<hash>/out/lib/; this script copies the most
// recently modified candidate into the package directory.

import { copyFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pkgDir = dirname(fileURLToPath(import.meta.url));
const targetDir = resolve(pkgDir, "../../target");

const candidates = [];
for (const profilePath of [
  targetDir,
  join(targetDir, "aarch64-apple-darwin"),
]) {
  for (const profile of ["release", "debug"]) {
    const buildDir = join(profilePath, profile, "build");
    if (!existsSync(buildDir)) {
      continue;
    }
    for (const entry of readdirSync(buildDir)) {
      if (!entry.startsWith("mlex-")) {
        continue;
      }
      const metallib = join(buildDir, entry, "out", "lib", "mlx.metallib");
      if (existsSync(metallib)) {
        candidates.push({ path: metallib, mtime: statSync(metallib).mtimeMs });
      }
    }
  }
}

if (candidates.length === 0) {
  console.error(
    "copy-metallib: no mlx.metallib found under the cargo target directory — " +
      "did the native build run?",
  );
  process.exit(1);
}

candidates.sort((a, b) => b.mtime - a.mtime);
const source = candidates[0].path;
const dest = join(pkgDir, "mlx.metallib");
copyFileSync(source, dest);
console.log(`copy-metallib: ${source} -> ${dest}`);
