use std::env;
use std::path::PathBuf;

fn main() {
    if std::env::var("VIRTUAL_ENV").is_err() {
        eprintln!("Source a virtualenv!");
        std::process::exit(1);
    }

    let normal_venv = PathBuf::from(&format!(
        "{}/lib/python3.13/site-packages/torch/lib",
        std::env::var("VIRTUAL_ENV").unwrap_or("".to_owned())
    ));
    let dev_venv = PathBuf::from(&format!(
        "{}/../build/lib/",
        std::env::var("VIRTUAL_ENV").unwrap_or("".to_owned())
    ));
    let lib_path = if normal_venv.is_dir() {
        normal_venv
    } else if dev_venv.is_dir() {
        dev_venv
    } else {
        eprintln!(
            "This really doesn't look like a venv I can use, or paths just dont work out, fixed on python3.13 atm."
        );
        std::process::exit(1);
    };
    //println!("cargo::rerun-if-changed=build.rs");
    // let lib_path = ;
    //let lib_path = "/workspace/ivor/ml/pytorch_dev/pytorch/build/lib";
    // let lib_path = "/home/ivor/Documents/Code/rust/overlay_segmenter/repo/train/.venv/lib/python3.13/site-packages/torch/lib/";
    // Tell cargo to look for shared libraries in the specified directory
    let lib_path = lib_path.display();
    println!("cargo:rustc-link-search={lib_path}");

    // Why do we need this? :/ Without it we miss stuff like undefined reference: std::__throw_bad_alloc()
    println!("cargo:rustc-link-lib=stdc++");

    let feature_cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    if feature_cuda {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-lib=torch_cuda");
    }
    println!("cargo:rustc-link-lib=torch_cpu");
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_cuda");
    //
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
    // println!("cargo:rustc-link-arg=-L{}", lib_path);
    //
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // println!("cargo:rustc-link-search=static:{}", out_path.display());
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_stable");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_consts.rs");
    let mut lines = String::new();
    lines += &format!(
        "/// This constant holds the library directory according to torch_stable's compilation.\n"
    );
    lines += &format!("pub const LIB_PATH: &str = \"{lib_path}\";\n");

    lines += &format!("/// This holds torch_stable's 'cuda' feature flag.\n");
    lines += &format!("pub const FEATURE_CUDA: bool = {feature_cuda};\n");
    std::fs::write(&dest_path, lines).unwrap();
}
