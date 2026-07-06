use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlx_c_dir = manifest_dir.join("vendor/mlx-c");
    let mlx_dir = manifest_dir.join("vendor/mlx");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!(
        "cargo:rerun-if-changed={}",
        mlx_c_dir.join("mlx/c").display()
    );

    let mut cfg = cmake::Config::new(&mlx_c_dir);
    cfg.define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        // Use the vendored MLX checkout instead of letting FetchContent clone it.
        .define("FETCHCONTENT_SOURCE_DIR_MLX", mlx_dir.to_str().unwrap())
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .profile("Release");
    if cfg!(target_os = "macos")
        && let Ok(target) = env::var("MACOSX_DEPLOYMENT_TARGET")
        && !target.trim().is_empty()
    {
        cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", target);
    }
    let dst = cfg.build();

    // mlx-c installs libmlxc; libmlx stays in the CMake build tree.
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    let build_dir = dst.join("build");
    for sub in ["_deps/mlx-build", "_deps/mlx-build/mlx", "lib"] {
        let p = build_dir.join(sub);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Locate the built mlx.metallib so dependents can colocate it next to
    // their final binary (MLX also bakes the absolute build path in, which
    // covers local development).
    if cfg!(target_os = "macos") {
        let candidates = [
            build_dir.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
            build_dir.join("_deps/mlx-build/mlx/backend/metal/mlx.metallib"),
        ];
        for c in &candidates {
            if c.exists() {
                println!("cargo:metallib={}", c.display());
                println!("cargo:rustc-env=MLEX_METALLIB_PATH={}", c.display());
                break;
            }
        }
    }

    let bindings = bindgen::Builder::default()
        .header(mlx_c_dir.join("mlx/c/mlx.h").to_str().unwrap().to_owned())
        .clang_arg(format!("-I{}", mlx_c_dir.display()))
        .allowlist_function("mlx_.*")
        .allowlist_function("_mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .layout_tests(false)
        .generate_comments(false)
        .derive_default(true)
        .generate()
        .expect("bindgen failed for mlx-c headers");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
