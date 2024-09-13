fn main() {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}