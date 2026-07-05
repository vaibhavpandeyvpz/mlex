# Vendored sources

These checkouts are vendored verbatim (with their own `.git` metadata
stripped) so `mlex` builds offline and is self-contained when published to
crates.io. `build.rs` builds them via CMake (`mlx-c`'s `CMakeLists.txt`
pulls in `mlx` through `FETCHCONTENT_SOURCE_DIR_MLX`, which we point at
`vendor/mlx`).

To refresh either checkout, re-clone at the desired ref, delete its `.git`
directory, and update the pin below.

| Directory      | Upstream                                | Pinned commit                              |
| -------------- | --------------------------------------- | ------------------------------------------ |
| `vendor/mlx`   | https://github.com/ml-explore/mlx.git   | `68cf2fddd8de5edd8ab3d926391772b2e2cedad8` |
| `vendor/mlx-c` | https://github.com/ml-explore/mlx-c.git | `fba4470b89073180056c9ea46c443051375f7399` |
