//! Binary entry point; all logic lives in the `glazer_lib` crate so it stays
//! testable. `windows_subsystem` keeps the console window hidden in release.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    glazer_lib::run()
}
