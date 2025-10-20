# SingN'Seek - AI Coding Instructions

## Critical Rules

1. **macOS Compatibility**: Always verify that any terminal commands are compatible with macOS/zsh before running them.

2. **Use `uv` Package Manager**: This project uses `uv` instead of `pip` for Python package management. Always use `uv` commands:
   - Install packages: `uv add <package>`
   - Sync dependencies: `uv sync`
   - Run scripts: `uv run <command>`
   - Never use `pip install` or `pip` commands

3. **Virtual Environment Check**: Always verify that the virtual environment (`.venv/`) is activated before running any Python commands. Check with `which python` or ensure commands are prefixed with `uv run`.

4. **Text-Only UI**: Never use icons in any generated code or UI elements unless explicitly requested. Always use descriptive text instead of icons for buttons, indicators, and interface elements.
