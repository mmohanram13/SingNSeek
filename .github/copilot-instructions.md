# SingN'Seek - AI Coding Instructions

## Critical Rules

1. **macOS Compatibility**: Always verify that any terminal commands are compatible with macOS/zsh before running them.

2. **Use `pip` Package Manager**: This project uses `pip` for Python package management. Always use `pip` commands:
   - Install packages: `pip install <package>`
   - Sync dependencies: `pip sync`
   - Run scripts: `pip run <command>`
   - Never use `uv` commands

3. **Virtual Environment Check**: Always verify that the virtual environment (`.venv/`) is activated before running any Python commands. Check with `which python` or ensure commands are prefixed with `pip run`.

4. **Text-Only UI**: Never use icons in any generated code or UI elements unless explicitly requested. Always use descriptive text instead of icons for buttons, indicators, and interface elements.
