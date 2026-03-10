# UV Setup Guide

This project supports [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver written in Rust.

## Installation

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via pip:**
```bash
pip install uv
```

## Quick Start

### 1. Install dependencies and create virtual environment
```bash
uv sync
```

This will create a virtual environment (`.venv/`) and install all dependencies. The project scripts are configured to be included in the build.

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies from `pyproject.toml`
- Generate `uv.lock` for reproducible builds

### 2. Activate the virtual environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### 3. Run the pipeline
```bash
python pipeline.py
```

## Common Commands

### Install dependencies
```bash
uv sync
```

**Alternative:** If you prefer to skip installing the project itself and only install dependencies, you can use:
```bash
uv sync --no-install-project
```

### Add a new dependency
```bash
uv add package-name
```

### Add a development dependency
```bash
uv add --dev package-name
```

### Remove a dependency
```bash
uv remove package-name
```

### Update all dependencies
```bash
uv sync --upgrade
```

### Run a command in the virtual environment
```bash
uv run python pipeline.py
```

### Run with specific Python version
```bash
uv python install 3.11
uv sync
```

## Benefits of uv

1. **Speed**: 10-100x faster than pip
2. **Reproducible builds**: `uv.lock` ensures consistent installs
3. **Better dependency resolution**: Handles complex dependency graphs
4. **Drop-in replacement**: Works with existing `pyproject.toml` files
5. **Virtual environment management**: Automatically creates and manages `.venv/`

## Migration from pip

If you're currently using `pip` and `requirements.txt`:

1. Your `pyproject.toml` is already compatible with uv
2. Simply run `uv sync` to create the virtual environment
3. The `requirements.txt` file is kept for compatibility but uv uses `pyproject.toml`

## Troubleshooting

### Virtual environment not found
```bash
uv sync  # This will create .venv/
```

### Lock file out of sync
```bash
uv lock  # Regenerate uv.lock
```

### Clear cache
```bash
uv cache clean
```

## More Information

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
