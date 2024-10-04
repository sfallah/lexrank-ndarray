fn main() {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    println!("cargo:rustc-link-lib=framework=Accelerate");
    #[cfg(all(target_os = "linux", feature = "blas"))]
    println!("cargo:rustc-link-lib=blas");
    #[cfg(all(target_os = "linux", feature = "mkl"))]
    println!("cargo:rustc-link-search=/opt/intel/mkl");
}
