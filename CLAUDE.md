# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning-powered taste prediction application built with Streamlit. It analyzes script content to predict which audience clusters (taste profiles) will likely adopt the content. The system uses:
- Groq LLM API for script analysis and metadata extraction
- Sentence transformers (all-MiniLM-L6-v2) for generating embeddings
- Pre-trained clustering models with 12 distinct taste clusters
- Random Forest regression for adoption probability prediction

## Development Commands

### Running the Application
```bash
streamlit run taste_predictor.py
```
The app runs on port 5000 by default (configured in `.streamlit/config.toml`).

### Installing Dependencies
```bash
pip install -r requirements.txt
```
Note: Uses PyTorch CPU-only version. The requirements file includes a custom PyPI index for CPU wheels.

### Syntax Checking
```bash
python -m py_compile taste_predictor.py
```
No formal test suite or linter is configured.

## Architecture

### Core Application Flow
1. **Script Upload & Metadata Generation** (`llm_define_metadata_v2`):
   - User uploads script (.txt file) via Streamlit file uploader
   - LLM makes two-stage analysis:
     - Stage 1: Extracts core fields (FRANCHISE_TITLE, logline, Genre, Subgenre, CONTENT_TYPE)
     - Stage 2: Extracts detailed fields (Tonal_Comps, Shared_Tropes, Differential, Protagonist_Demo)
   - Creates `embedding_text` by concatenating all metadata fields

2. **Prediction Pipeline** (`predict_from_metadata`):
   - Generates normalized embedding from metadata text using SentenceTransformer
   - For each of 12 clusters:
     - Concatenates content embedding with cluster centroid (Path B architecture)
     - Runs through pre-trained Random Forest model to predict adoption probability
   - Returns sorted DataFrame with cluster predictions

3. **Visualization**:
   - Displays results in 3x4 grid showing adoption probability for each cluster
   - Shows full dataframe with cluster names and scores

### Key Data Structures
- **CLUSTER_LABELS**: Hardcoded dict mapping cluster IDs (0-11) to human-readable names (e.g., "romance_thriller_darkcomedy", "animated_superhero")
- **trained_models.pkl**: Contains pre-trained models, specifically `path_b_rfr` (Random Forest Regressor)
- **centroids.pkl**: Contains normalized centroid vectors for each of 12 clusters
- **cluster_summary.csv**: Contains cluster metadata (cluster_id, cluster_name, users count)

### Model Caching
All models and data are loaded once via `@st.cache_resource` decorator on `load_models()` function. This includes:
- SentenceTransformer embedder
- Trained ML models from joblib
- Cluster centroids
- Cluster summary DataFrame

### Groq LLM Integration
- Uses `_call_llm()` wrapper function for all LLM calls
- API key loaded from `.env` file via python-dotenv
- Default model: `openai/gpt-oss-120b` (configurable in UI)
- LLM returns must be valid JSON (prompts specify this requirement)

## Code Style & Conventions

### Imports
- Standard library first (os, json)
- Third-party second (streamlit, pandas, groq)
- Use `import X as Y` for common modules (e.g., `st`, `pd`, `np`)

### Naming
- Functions: `snake_case`
- Variables: `snake_case`
- Constants: `UPPER_CASE`
- Type hints on function parameters

### Error Handling
- Graceful degradation for optional dependencies (sentence-transformers wrapped in try/except)
- User-facing errors displayed with `st.error()` or `st.warning()`
- Functions return `None` or empty structures on failure

### Formatting
- F-strings for string formatting
- `textwrap.dedent()` for multi-line prompt strings
- 4-space indentation
- **No emojis** - Keep UI professional and text-only

## Session State Management
The app uses Streamlit session state to persist data between reruns:
- `st.session_state.metadata`: Stores generated/edited metadata dict
- `st.session_state.predictions`: Stores prediction results DataFrame

## Environment Variables
Required in `.env` file:
- `GROQ_API_KEY`: API key for Groq LLM service

## Deployment
Configured for Replit deployment with:
- Autoscale deployment target
- Streamlit runs on port 5000 (internal), exposed on port 80 (external)
- Alternative port 8501 exposed on port 3000