use std::env;
use std::path::PathBuf;
use std::process::Command;

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
    if cfg!(target_os = "macos") {
        if let Ok(target) = env::var("MACOSX_DEPLOYMENT_TARGET") {
            if !target.trim().is_empty() {
                cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", target);
            }
        }
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
        link_clang_rt_builtins();
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

/// MLX's C++ uses `__builtin_available`, which clang lowers to a call to
/// `___isPlatformVersionAtLeast` from compiler-rt's builtins. Apple clang
/// links `libclang_rt.osx.a` implicitly, but rustc drives the final link
/// with its own `compiler_builtins` (which lacks that symbol), so we must
/// link clang's builtins archive explicitly.
fn link_clang_rt_builtins() {
    let clang = env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let output = Command::new(&clang)
        .arg("--print-resource-dir")
        .output()
        .expect("failed to run clang --print-resource-dir");
    if !output.status.success() {
        panic!(
            "clang --print-resource-dir failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let resource_dir = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
    let darwin_lib_dir = resource_dir.join("lib/darwin");
    let lib = darwin_lib_dir.join("libclang_rt.osx.a");
    if !lib.exists() {
        panic!(
            "libclang_rt.osx.a not found in {} (needed for ___isPlatformVersionAtLeast)",
            darwin_lib_dir.display()
        );
    }
    println!(
        "cargo:rustc-link-search=native={}",
        darwin_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
