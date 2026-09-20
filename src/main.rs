fn greeting() -> String {
    "nano-vllm-rs scaffold is alive".to_string()
}

fn main() {
    println!("{}", greeting());
}

#[cfg(test)]
mod tests {
    use super::greeting;

    #[test]
    fn scaffold_is_alive() {
        assert!(greeting().contains("alive"));
    }
}
