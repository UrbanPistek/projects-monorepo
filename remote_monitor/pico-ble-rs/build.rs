//! build.rs — emits the `memory.x` linker script search path.
//!
//! The `cortex-m-rt` crate looks for `memory.x` via a linker search path flag.
//! Without this script, the linker cannot find the file even if it sits in the
//! project root.  This is the standard pattern used by all Embassy RP examples.

use std::{env, fs, path::PathBuf};

fn main() {
    // Tell Cargo to re-run this script if memory.x changes.
    println!("cargo:rerun-if-changed=memory.x");

    // Copy memory.x into the build output directory, then add that directory
    // to the linker search path with `-L`.
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
}
