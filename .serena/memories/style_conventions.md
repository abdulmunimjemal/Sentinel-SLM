# Style and Conventions

- Use Python type hints (`typing`) in all functions.
- Use `logging` instead of `print` for runtime output.
- Use `os.path.join` with relative paths for filesystem paths.
- Rail A label convention: `0 = Safe`, `1 = Attack`.
- Taxonomy source of truth: `src/sentinel/utils/taxonomy.py` (categories 0-8; 8 is PROMPT_ATTACK).
