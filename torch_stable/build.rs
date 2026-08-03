use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild if a different virtualenvironment is sourced.
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");

    if std::env::var("VIRTUAL_ENV").is_err() {
        return Err("VIRTUAL_ENV was not set, source a virtual environment before building".into());
    }

    let env_var = std::env::var("VIRTUAL_ENV")?;

    // Execute the python3 command to print the version string, like '3.13'
    let python_result = std::process::Command::new("python")
        .args([
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}', end='')",
        ])
        .output()?;
    if !python_result.status.success() {
        return Err("could not run python3 to retrieve the python version".into());
    }
    let python_tag = String::from_utf8(python_result.stdout)?;

    // Next, we want to handle two types of env vars, the normal one everyone uses:
    let normal_venv = PathBuf::from(&format!(
        "{env_var}/lib/python{python_tag}/site-packages/torch/lib",
    ));
    // And the not normal one that those building PyTorch from source can use;
    let dev_venv = PathBuf::from(&format!("{env_var}/../build/lib/",));

    // Then try to find the normal one, fallback to dev.
    let lib_path = if normal_venv.is_dir() {
        normal_venv
    } else if dev_venv.is_dir() {
        dev_venv
    } else {
        return Err(
            format!("couldn't library dirs, tried {normal_venv:?} and {dev_venv:?}").into(),
        );
    };

    // Next, add the search path.
    let lib_path = lib_path.display();
    println!("cargo:rustc-link-search={lib_path}");

    // Why do we need this? :/ Without it we miss stuff like undefined reference: std::__throw_bad_alloc()
    println!("cargo:rustc-link-lib=stdc++");

    // If building with cuda, also link with cuda.
    let feature_cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    if feature_cuda {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-lib=torch_cuda");
    }
    // Always link against cpu
    println!("cargo:rustc-link-lib=torch_cpu");
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_cuda");

    // And against normal torch.
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");

    // And set the runpath, such that we don't need to muck with LD_LIBRARY_PATH
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);

    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // println!("cargo:rustc-link-search=static:{}", out_path.display());
    // println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=torch_stable");

    // Next, we're going to write our information about where the library is located for downstream consumers.
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_consts.rs");
    let mut lines = String::new();
    lines +=
        "/// This constant holds the library directory according to torch_stable's compilation.\n";
    lines += &format!("pub const LIB_PATH: &str = \"{lib_path}\";\n");

    lines += "/// This holds torch_stable's 'cuda' feature flag.\n";
    lines += &format!("pub const FEATURE_CUDA: bool = {feature_cuda};\n");
    std::fs::write(&dest_path, lines).unwrap();

    Ok(())
}
