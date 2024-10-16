# Efficient LexRank for Extractive Summarization

This implementation of LexRank focuses on efficient extractive summarization based on **sentence centrality**, utilizing **ndarray** for performance optimization.

## Usage

### Benchmarks
Run benchmarks using the following command:
```bash
cargo bench --features testing
```

### Tests
Execute tests with:
```bash
cargo test --test tests --features testing
```

### Python Library
To create the Python library, run:
```bash
maturin develop --release --features extension-module
```

After that feel free to check it if runs correctly with

```bash
python tests/test_bindings.py 
```