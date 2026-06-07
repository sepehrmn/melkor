# Contributing

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the build and tests
5. Submit a pull request

## Development Setup

```bash
# Build the converter
./scripts/setup_deps.sh
mkdir -p build && cd build
cmake .. && make -j$(sysctl -n hw.ncpu)
```

## Code Style

- C++17 with 4-space indentation
- Python with ruff defaults
- Shell scripts with 2-space indentation

## Pull Request Checklist

- [ ] Code builds and runs
- [ ] No new warnings
- [ ] Python code passes `ruff check`
