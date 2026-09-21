//! Integration tests: drive the compiled binary exactly like a user would.
//! Cargo exposes the binary's path as CARGO_BIN_EXE_<name> to these tests.

use std::process::{Command, Output};

fn run_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nano-vllm-rs"))
        .args(args)
        .output()
        .expect("the course binary should run")
}

#[test]
fn generate_cli_is_deterministic() {
    let first = run_cli(&["generate", "ru", "12"]);
    let second = run_cli(&["generate", "ru", "12"]);
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
    let text = String::from_utf8(first.stdout).expect("stdout is text");
    assert_eq!(text.trim(), "\"rust. rust. ru\"");
}

#[test]
fn missing_subcommand_exits_nonzero_with_usage() {
    let out = run_cli(&[]);
    assert!(!out.status.success());
    let usage = String::from_utf8(out.stderr).expect("stderr is text");
    assert!(usage.contains("usage:"));
}
