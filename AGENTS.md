# Agent Guidelines for Taste Predictor Project

## Build/Run/Test Commands
- **Run app**: `streamlit run taste_predictor.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Run single test**: No formal test suite; manually test via Streamlit interface
- **Lint**: No linter configured; use `python -m py_compile taste_predictor.py` for syntax check

## Code Style Guidelines

### Imports
- Standard library imports first (e.g., `import os`, `import json`)
- Third-party imports second (e.g., `import streamlit as st`, `import pandas as pd`)
- Local imports last (none in current codebase)
- Use `import X as Y` for commonly used modules (e.g., `import streamlit as st`)

### Naming Conventions
- **Functions**: snake_case (e.g., `predict_from_metadata`, `llm_define_metadata_v2`)
- **Variables**: snake_case (e.g., `script_text`, `content_type`)
- **Constants**: UPPER_CASE (e.g., `CLUSTER_LABELS`, `HAS_SENTENCE_TRANSFORMERS`)
- **Classes**: PascalCase (when used)

### Formatting & Style
- Use type hints for function parameters (e.g., `prompt: str`, `groq_model: str`)
- Use f-strings for string formatting (e.g., `f"Error: {e}"`)
- Use `textwrap.dedent()` for multi-line strings in prompts
- Line length: Keep under 120 characters when possible
- Use 4 spaces for indentation (Python standard)

### Error Handling
- Use try/except blocks for external API calls and file operations
- Display user-friendly error messages with `st.error()`
- Graceful degradation for optional dependencies (e.g., sentence-transformers)
- Return `None` or empty structures on failure rather than crashing

### Documentation
- Add docstrings for complex functions explaining purpose and parameters
- Use inline comments for non-obvious logic
- Keep code self-documenting with descriptive variable names

### Best Practices
- Cache expensive operations with `@st.cache_resource`
- Handle optional dependencies with try/except imports
- Use environment variables for API keys (via python-dotenv)
- Validate inputs before processing
- Prefer functional programming patterns over side effects