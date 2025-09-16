import os
import json
import importlib
from typing import Dict, Optional, Tuple, List, Any
from textwrap import dedent
from io import StringIO

import numpy as np

# Only run Streamlit app if dependencies are available
try:
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
    from groq import Groq
    from script_to_metadata import ScriptMetadataExtractor
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Required dependencies not available: {e}")
    print("Please install: pandas, streamlit, plotly, groq")
    exit(1)

# Optional Groq LLM
try:
    from groq import Groq  # type: ignore
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False


# -----------------------
# LLM extraction helpers
# -----------------------
def _call_llm(prompt: str, groq_model: str) -> Optional[str]:
    if not HAS_GROQ:
        return None
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=groq_model,
        )
        return chat_completion.choices[0].message.content
    except Exception:
        return None


def _load_example_rows(n: int = 5) -> List[Dict]:
    """Load a few example rows from franchise_metadata.csv to include in prompts."""
    candidates = [
        'franchise_metadata.csv',
        os.path.join('notebooks', 'outputs', 'franchise_metadata.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                cols = ['FRANCHISE_TITLE', 'Genre', 'Subgenre', 'CONTENT_TYPE']
                df = df[[c for c in cols if c in df.columns]].dropna(how='all')
                return df.head(n).to_dict(orient='records')
            except Exception:
                continue
    return []


def llm_define_metadata_v2(script_text: str, content_type: str, groq_model: str) -> Optional[Dict]:
    """Replicate the notebook's 2-stage LLM extraction with examples and guidance."""
    # Build examples text like the notebook
    sample_examples = _load_example_rows(6)
    examples_text = ""
    if sample_examples:
        examples_text = "\n\nEXAMPLES from franchise_metadata.csv:\n" + "\n".join([
            f"- {ex.get('FRANCHISE_TITLE','N/A')} | Genre: {ex.get('Genre','N/A')} | Subgenre: {ex.get('Subgenre','N/A')} | Content Type: {ex.get('CONTENT_TYPE','N/A')}"
            for ex in sample_examples
        ])
    summary_prompt = dedent(f"""
        You are a professional script reader for a major studio.
        Your task is to summarize the following script in 3-4 concise sentences.
        Focus on the main plot, primary characters, setting, and overall tone.
        Do not add any preamble or explanation. Provide only the summary.

        SCRIPT TEXT:
        ---
        {script_text[:2000]}
        ---
    """)
    summary = _call_llm(summary_prompt, groq_model)
    if not summary:
        return None

    core_fields = ['FRANCHISE_TITLE', 'logline', 'Genre', 'Subgenre', 'CONTENT_TYPE']
    core_prompt = dedent(f"""
        You are a professional script analyst and development executive. Your task is to read a script summary and extract its core creative DNA.

        TASK:
        Analyze the provided script summary and identify its core attributes.

        OUTPUT FORMAT:
        Your response MUST be a single, valid JSON object. Do not include any other text, preamble, or explanation.
        The JSON object must contain ONLY the following keys: {json.dumps(core_fields)}

        EXAMPLES:
        {examples_text}

        SCRIPT SUMMARY:
        ---
        {summary}
        ---

        Now, provide the JSON output.
    """)
    try:
        core_attrs_str = _call_llm(core_prompt, groq_model)
        if core_attrs_str is None:
            return None
        core_attrs = json.loads(core_attrs_str)
    except Exception:
        return None

    detailed_fields = ['Tonal_Comps', 'Shared_Tropes', 'Differential', 'Protagonist_Demo']
    detailed_prompt = dedent(f"""
        You are a professional script analyst and development executive. Your task is to read a script summary and its core attributes, then extract detailed descriptive elements.

        TASK:
        Analyze the provided summary and core attributes to identify deeper creative and demographic details.
        - For 'Tonal_Comps', list 3-5 existing movies or TV shows with a similar tone and feel.
        - For 'Shared_Tropes', identify 3-5 common narrative tropes or themes present in the story.
        - For 'Differential', explain in one sentence what makes this concept unique or fresh.
        - For 'Protagonist_Demo', describe the main character(s) including age, gender, and profession if known.

        OUTPUT FORMAT:
        Your response MUST be a single, valid JSON object. Do not include any other text, preamble, or explanation.
        The JSON object must contain ONLY the following keys: {json.dumps(detailed_fields)}

        The movie Friday is a comedy where scenes that would normally be scary or sad are presented in a light-hearted way. When building descriptions make sure to emphasize the comedic elements instead of the darker themes. 

        EXAMPLES:
        {examples_text}

        SCRIPT DETAILS:
        ---
        Summary: {summary}
        Core Attributes: {json.dumps(core_attrs)}
        ---

        Now, provide the JSON output.
    """)
    try:
        detailed_attrs_str = _call_llm(detailed_prompt, groq_model)
        if detailed_attrs_str is None:
            return None
        detailed_attrs = json.loads(detailed_attrs_str)
    except Exception:
        return None

    metadata = {**core_attrs, **detailed_attrs}
    metadata['CONTENT_TYPE'] = metadata.get('CONTENT_TYPE') or content_type
    metadata['embedding_text'] = (
        f"{metadata.get('FRANCHISE_TITLE','')} | Genre: {metadata.get('Genre','')} | Sub: {metadata.get('Subgenre','')} | "
        f"Tonal: {metadata.get('Tonal_Comps','')} | Tropes: {metadata.get('Shared_Tropes','')} | "
        f"Diff: {metadata.get('Differential','')} | Protagonist: {metadata.get('Protagonist_Demo','')}"
    )
    return metadata


def extract_labels(script_text: str, default_type: str = "TV Show") -> Dict:
    """Attempt Groq LLM extraction identical to notebook; fallback to rule-based extractor."""
    groq_model = os.environ.get('GROQ_MODEL_NAME', 'kimi-k2-instruct')
    if HAS_GROQ and os.getenv('GROQ_API_KEY'):
        meta = llm_define_metadata_v2(script_text, default_type, groq_model)
        if meta:
            return meta
    # Fallback – rule-based extractor
    extractor = ScriptMetadataExtractor()
    return extractor.extract_metadata(script_text, content_type=default_type)


# -----------------------
# Prediction helpers
# -----------------------
class _SyntheticPredictor:
    def __init__(self):
        self.cluster_info = {
            0: {'name': 'Casual & Fading'},
            1: {'name': 'Diverse Dabblers'},
            2: {'name': 'Low-Engagement Actives'},
            3: {'name': 'Superfans'},
            4: {'name': 'Power Users'},
            5: {'name': 'Franchise Fans'},
            6: {'name': 'Ephemeral Bingers'},
            7: {'name': 'Balanced Viewers'},
        }
        self.is_synthetic = True

    def get_cluster_size(self, cluster_id: int) -> int:
        cluster_sizes = {0: 5346, 1: 250, 2: 1898, 3: 21, 4: 133, 5: 602, 6: 1297, 7: 453}
        return cluster_sizes.get(cluster_id, 1000)

    def predict_engagement(self, content_input: Dict) -> Dict:
        genre = content_input.get('genre', 'Drama')
        content_type = content_input.get('content_type', 'Movie')
        genre_patterns = {
            'Comedy': {0: 0.05, 1: 0.35, 2: 0.08, 3: 0.65, 4: 0.45, 5: 0.15, 6: 0.12, 7: 0.18},
            'Drama': {0: 0.02, 1: 0.25, 2: 0.06, 3: 0.78, 4: 0.52, 5: 0.20, 6: 0.08, 7: 0.15},
            'Horror': {0: 0.01, 1: 0.40, 2: 0.05, 3: 0.45, 4: 0.35, 5: 0.12, 6: 0.18, 7: 0.10},
            'Reality': {0: 0.08, 1: 0.15, 2: 0.12, 3: 0.25, 4: 0.20, 5: 0.08, 6: 0.25, 7: 0.30},
        }
        base_pattern = genre_patterns.get(genre, genre_patterns['Drama'])
        type_multiplier = 1.2 if content_type == 'TV Show' else 1.0
        np.random.seed(hash(str(content_input)) % 2**32)
        results = {}
        for cluster_id in range(8):
            cluster_name = self.cluster_info[cluster_id]['name']
            base_adoption = base_pattern[cluster_id] * type_multiplier
            variance = np.random.normal(0, base_adoption * 0.2)
            adoption_rate = max(0.001, min(0.95, base_adoption + variance))
            completion_rate = adoption_rate * np.random.uniform(0.6, 1.0)
            engagement_score = adoption_rate * np.random.uniform(0.1, 0.3)
            results[cluster_id] = {
                'cluster_name': cluster_name,
                'adoption_rate': adoption_rate,
                'completion_rate': completion_rate,
                'engagement_score': engagement_score,
                'predicted_viewers': int(adoption_rate * self.get_cluster_size(cluster_id)),
                'engagement_level': 'High' if engagement_score >= 0.15 else ('Medium' if engagement_score >= 0.08 else ('Low' if engagement_score >= 0.03 else 'Very Low'))
            }
        return results


def get_predictor():
    """Return a real predictor if TensorFlow + artifacts are available; else synthetic."""
    try:
        ep_mod = importlib.import_module('enhanced_prediction_dashboard')
        EnhancedContentPredictor = getattr(ep_mod, 'EnhancedContentPredictor', None)
    except Exception:
        EnhancedContentPredictor = None

    if EnhancedContentPredictor is not None:
        try:
            return EnhancedContentPredictor('simplified_enhanced_taste_model.keras', 'simplified_enhanced_preprocessor.pkl')
        except Exception:
            # Try synthetic mode within enhanced predictor
            try:
                return EnhancedContentPredictor('', '')
            except Exception:
                pass
    # Fallback synthetic predictor
    return _SyntheticPredictor()


def metadata_to_predictor_input(metadata: Dict) -> Dict:
    comps = [s.strip() for s in str(metadata.get('Tonal_Comps', '')).split(',') if s.strip()]
    tropes = [s.strip() for s in str(metadata.get('Shared_Tropes', '')).split(',') if s.strip()]
    return {
        'title': metadata.get('FRANCHISE_TITLE', 'Untitled'),
        'content_type': metadata.get('CONTENT_TYPE', 'TV Show'),
        'genre': metadata.get('Genre', 'Drama'),
        'subgenre': metadata.get('Subgenre', ''),
        'critic_score': 0,  # Optional fields not present in labels
        'audience_score': 0,
        'tonal_comparisons': comps,
        'shared_tropes': tropes,
    }


def build_adoption_heatmap(predictions: Dict[int, Dict]) -> go.Figure:
    # Order clusters by id and fill into 3x4 grid
    ids = sorted(predictions.keys())
    rows, cols = 3, 4
    z = [[np.nan for _ in range(cols)] for _ in range(rows)]
    text = [["" for _ in range(cols)] for _ in range(rows)]

    for idx, cid in enumerate(ids):
        r, c = divmod(idx, cols)
        if r >= rows:
            break
        p = predictions[cid]
        percent = p['adoption_rate'] * 100.0
        z[r][c] = p['adoption_rate']
        text[r][c] = f"{p['cluster_name']}<br>{percent:.1f}%"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, '#f0f0f0'], [0.5, '#d0d0d0'], [1.0, '#b0b0b0']],
            zmin=0, zmax=1,
            colorbar=dict(title="Adoption")
        )
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(
    page_title="BET+ Taste Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with glassmorphism and premium effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force light mode by default regardless of system settings */
    .stApp {
        color-scheme: light !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Force light mode on all elements */
    * {
        color-scheme: light !important;
    }
    
    /* Override system dark mode preferences */
    @media (prefers-color-scheme: dark) {
        .stApp {
            color-scheme: light !important;
            background: #ffffff !important;
            color: #000000 !important;
        }
        
        * {
            color-scheme: light !important;
        }
    }
    
    :root { --border-color: #e5e7eb; --transition-base: none; --text-primary: #000000; --text-secondary: #666666; --primary-color: #1f2937; --primary-hover: #374151; --primary-light: #ffffff; --accent-color: #1f2937; --success-color: #16a34a; }
    
    /* Override Streamlit's default dark mode handling */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Enforce plain white background and black text */
    .stApp {
        background: #ffffff !important;
        min-height: 100vh;
        color: #000000 !important;
        position: relative;
        animation: none !important;
        background-image: none !important;
    }

    .stApp::before { display: none !important; }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Force light backgrounds on all Streamlit containers */
    .main .block-container {
        background-color: transparent !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"]) {
        background-color: transparent !important;
    }

    /* Main container with enhanced spacing */
    .block-container {
        padding-top: 2rem;
        max-width: 1280px;
        animation: fadeInUp 0.6s ease-out;
        border: none !important;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Simplified top bar with white bg */
    .topbar {
        position: sticky;
        top: 1rem;
        z-index: 50;
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 0;
        background: #ffffff !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
        border: none !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: none;
        color: #000000 !important;
    }
    
    .topbar:hover {
        box-shadow: 0 8px 40px rgba(31, 38, 135, 0.25);
        transform: translateY(-2px);
    }
    .brand {
        display: inline-flex;
        gap: 14px;
        align-items: center;
        font-weight: 600;
        color: #000000 !important;
        font-size: 1.25rem;
    }
    .brand .logo {
        width: 42px;
        height: 42px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: #ffffff !important;
        color: #000000 !important;
        font-size: 1.5rem;
        box-shadow: none !important;
        transition: none;
    }
    .brand .logo:hover {
        transform: rotate(-5deg) scale(1.1);
    }
    .brand .beta {
        color: #000000 !important;
        background: #ffffff !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        box-shadow: none !important;
    }

    /* Enhanced hero with animated background */
    .main-header {
        text-align: center;
        padding: 2rem;
        border-radius: 0;
        background: #ffffff !important;
        box-shadow: none !important;
        margin-bottom: 1rem;
        position: relative;
        overflow: visible;
        animation: none;
        color: #000000 !important;
        border: none !important;
    }
    .main-header::before, .main-header::after { display: none !important; }
    @keyframes backgroundScroll {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem;
        color: #000000 !important;
        letter-spacing: -0.02em;
        position: relative;
        text-shadow: none;
        animation: none;
    }
    .main-header .subtitle {
        color: #000000 !important;
        max-width: 700px;
        margin: 0 auto;
        font-size: 1.25rem;
        font-weight: 300;
        position: relative;
        line-height: 1.6;
        animation: none;
    }
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Glass morphism cards */
    .card {
        background: #ffffff !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border-radius: 8px;
        padding: 1rem;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 1rem;
        transition: none;
        position: relative;
        overflow: visible;
        color: #000000 !important;
    }
    .card::before { display: none !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff !important;
        gap: 0.5rem;
        border-bottom: none !important;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #000000 !important;
        background: transparent !important;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000 !important;
        background: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-active"] {
        color: #000000 !important;
        background: #ffffff !important;
        border-bottom: 2px solid #000000 !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: none !important;
        padding: 0.625rem 0.875rem;
        background: #ffffff !important;
        color: #000000 !important;
        transition: none;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #000000 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Fix for dark mode text in inputs */
    .stTextInput input,
    .stTextArea textarea {
        -webkit-text-fill-color: var(--text-primary) !important;
    }

    /* Enhanced buttons with gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.025em;
        box-shadow: 0 4px 15px rgba(31, 41, 55, 0.3);
        transition: var(--transition-base);
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(31, 41, 55, 0.4);
    }
    .stButton > button:hover::before {
        left: 100%;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Enhanced metrics with glass effect */
    div[data-testid="metric-container"] {
        background: #ffffff !important;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        transition: var(--transition-base);
        position: relative;
        overflow: hidden;
    }
    div[data-testid="metric-container"]::before { display: none !important; }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
    }
    div[data-testid="metric-container"]:hover::before {
        opacity: 0.1;
    }
    div[data-testid="metric-container"] label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        margin-bottom: 0.5rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        background: #ffffff !important;
        -webkit-background-clip: border-box;
        -webkit-text-fill-color: #000000 !important;
        background-clip: border-box;
        font-weight: 700 !important;
        font-size: 1.875rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }

    /* Neutral chips */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        background: #ffffff !important;
        color: #000000 !important;
        font-weight: 600;
        font-size: 0.875rem;
        border: 1px solid #e5e7eb !important;
        transition: none;
    }
    .chip:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(31, 41, 55, 0.2);
    }
    .chip.success, .chip.warn, .chip.info { background: #ffffff !important; color: #000000 !important; border-color: #e5e7eb !important; }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
    }
    
    /* Fix file uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #000000 !important;
    }
    
    /* Fix all button colors */
    .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    
    .stButton > button:hover {
        background: #f3f4f6 !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    
    .stButton > button[kind="primary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #f3f4f6 !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    /* File uploader browse button */
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e7eb !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #f3f4f6 !important;
        border: 1px solid #000000 !important;
    }

    /* Fix text colors throughout */
    .stMarkdown, .stText { color: #000000 !important; }
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; }
    p { color: #000000 !important; }

    /* Additional overrides for all Streamlit elements */
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
    }
    
    /* Fix selectbox dropdown options - more specific targeting */
    .stSelectbox > div > div > div > div {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
    }
    
    [data-baseweb="select"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }
    
    [data-baseweb="popover"] > div {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
    }
    
    [role="listbox"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
    }
    
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Override any remaining dark backgrounds */
    div, section, main {
        background-color: transparent !important;
    }
    
    /* Ensure all text is black */
    span, p, div, label, button {
        color: #000000 !important;
    }
    
    /* Override any remaining button styles */
    button, input[type="button"], input[type="submit"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e7eb !important;
    }
    
    button:hover, input[type="button"]:hover, input[type="submit"]:hover {
        background: #f3f4f6 !important;
        color: #000000 !important;
    }
    
    /* Very specific overrides for stubborn elements */
    [class*="uploadButton"], [class*="browse"], button[kind="secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* Footer removed */
    .app-footer { display: none !important; }
    
    /* Loading animation */
    .stSpinner > div { border-color: #000000 !important; }
    
    /* Success/Error messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px);
    }

    /* Remove unnecessary borders and shadows globally for a cleaner UI */
    /* This overrides any remaining border or box-shadow styles on common Streamlit elements. */
    .stApp, .block-container, .card, div[data-testid="metric-container"], .stDataFrame,
    section[data-testid="stSidebar"] > div, .chip, .stButton > button, .stButton > button[kind="primary"],
    [data-baseweb="popover"] > div, [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stExpander"], button, input[type="button"], input[type="submit"],
    [class*="uploadButton"], [class*="browse"], button[kind="secondary"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* Ensure sidebar divider is removed */
    section[data-testid="stSidebar"] > div {
        border-right: none !important;
    }

</style>
""", unsafe_allow_html=True)

# Top bar and hero with clean design
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div style="font-weight: 600; color: black; font-size: 18px;">BET+ Taste Predictor</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
        </div>
    </div>
    <div class="main-header">
        <h1>Predict Content Performance</h1>
        <p class="subtitle">Script analysis and audience taste prediction platform.<br>Transform your content strategy with data driven insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
    )

# Sidebar: quick actions and downloads
def _safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", " ")).strip().replace(" ", "_") or "untitled"

def _reset_session():
    for k in ["script_text", "labels", "predictions", "predictor"]:
        if k in st.session_state:
            del st.session_state[k]

with st.sidebar:
    st.markdown("""
    <div class='card' style='margin-top:0; background: #ffffff; border: 1px solid #e5e7eb;'>
        <h4 style='margin:0 0 1rem 0; color: #000000; font-weight: 600;'>Quick Actions</h4>
        <ul style='list-style: none; padding: 0; line-height: 2;'>
            <li>Label scripts automatically</li>
            <li>Predict audience clusters</li>
            <li>Analyze engagement metrics</li>
            <li>Export detailed reports</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset", use_container_width=True):
            _reset_session()
            st.success("Session reset!")
    with col2:
        if st.button("Help", use_container_width=True):
            st.info("Contact support")
    labels_ss = st.session_state.get("labels")
    preds_ss = st.session_state.get("predictions")
    if labels_ss:
        title = labels_ss.get("FRANCHISE_TITLE", "Untitled")
        st.download_button(
            "Download Labels JSON",
            json.dumps(labels_ss, indent=2),
            file_name=f"{_safe_filename(title)}_labels.json",
            mime="application/json",
        )
    if preds_ss:
        rows = []
        for cid, p in preds_ss.items():
            rows.append({
                "cluster_id": cid,
                "cluster": p.get("cluster_name"),
                "adoption": p.get("adoption_rate"),
                "completion": p.get("completion_rate"),
                "engagement": p.get("engagement_score"),
                "viewers": p.get("predicted_viewers"),
                "level": p.get("engagement_level"),
            })
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name=f"{_safe_filename(labels_ss.get('FRANCHISE_TITLE','untitled'))}_predictions.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)



def show_definitions_panel():
    with st.expander("Field Definitions Guide", expanded=False):
        st.markdown("""
        <div class="card" style="background: #ffffff; border: 1px solid #e5e7eb;">
            <h4 style="margin-top: 0; color: #000000; font-weight: 700;">Content Metadata Fields</h4>
            <div style="display: grid; gap: 0.75rem; margin-top: 1rem;">
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">FRANCHISE_TITLE</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">Working title or project name</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">CONTENT_TYPE</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">Format: TV Show or Movie</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">Genre / Subgenre</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">Primary and secondary genre classifications</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">Tonal_Comps</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">3-5 similar titles for tone reference</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">Shared_Tropes</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">3-5 narrative themes or story elements</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">Differential</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">What makes this concept unique</span>
                </div>
                <div style="padding: 0.75rem; background: #ffffff; border-radius: 8px; border-left: 3px solid #000000;">
                    <strong style="color: #000000;">Protagonist_Demo</strong>
                    <span style="color: #000000; display: block; margin-top: 0.25rem;">Main character demographics</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


tab_label, tab_predict, tab_roi = st.tabs([
    "Label Script", 
    "Taste Prediction", 
    "ROI Optimizer (Beta)"
])


# -----------------------
# Tab: Label Script
# -----------------------
with tab_label:
    st.markdown("""
    <div class=\"card\" style=\"background: #ffffff; border: 1px solid #e5e7eb;\">
        <div style=\"display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;\">
            <div>
                <h3 style=\"margin: 0; color: #000000; font-weight: 700;\">Step 1: Input Your Content</h3>
                <p style=\"color: #000000; margin: 0.25rem 0 0 0;\">Upload a script file or paste your content for analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("", type=["txt", "md"], accept_multiple_files=False)
    with col2:
        default_type = st.selectbox("Content Type", ["TV Show", "Movie"], index=0)
    
    script_text = st.text_area("Or paste script/brief here", height=200, placeholder="Paste your script content, synopsis, or brief here...")

    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8", errors="ignore")
            script_text = content
            st.success("File uploaded successfully!")
        except Exception:
            st.error("Could not read uploaded file.")

    show_definitions_panel()

    if st.button("Extract Labels", type="primary", disabled=(not script_text.strip()), use_container_width=True):
        with st.spinner("Analyzing your content..."):
            meta = extract_labels(script_text, default_type=default_type)
            st.session_state['script_text'] = script_text
            st.session_state['labels'] = meta
            st.success("Labels extracted successfully!")

    labels = st.session_state.get('labels')
    if labels:
        st.markdown("""
        <div class=\"card\" style=\"background: #ffffff; border: 1px solid #e5e7eb;\">
            <div style=\"display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;\">
                <div>
                    <h3 style=\"margin: 0; color: #000000; font-weight: 700;\">Step 2: Review & Refine</h3>
                    <p style=\"color: #000000; margin: 0.25rem 0 0 0;\">Fine-tune the AI-extracted labels for maximum accuracy</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Core Information")
            labels['FRANCHISE_TITLE'] = st.text_input("FRANCHISE_TITLE", labels.get('FRANCHISE_TITLE', ''))
            labels['CONTENT_TYPE'] = st.selectbox("CONTENT_TYPE", ["TV Show", "Movie"], index=0 if labels.get('CONTENT_TYPE','TV Show')=="TV Show" else 1)
            labels['Genre'] = st.text_input("Genre", labels.get('Genre', 'Drama'))
            labels['Subgenre'] = st.text_input("Subgenre", labels.get('Subgenre', ''))
            labels['Sub_sub_genre'] = st.text_input("Sub_sub_genre", labels.get('Sub_sub_genre', '') or '')
        with col2:
            st.markdown("#### Creative Elements")
            labels['Tonal_Comps'] = st.text_input("Tonal_Comps (comma-separated)", labels.get('Tonal_Comps', ''))
            labels['Shared_Tropes'] = st.text_input("Shared_Tropes (comma-separated)", labels.get('Shared_Tropes', ''))
            labels['Differential'] = st.text_area("Differential", labels.get('Differential', ''))
            labels['Protagonist_Demo'] = st.text_input("Protagonist_Demo", labels.get('Protagonist_Demo', ''))
            labels['logline'] = st.text_input("logline", labels.get('logline', ''))

        st.session_state['labels'] = labels
        st.markdown("""
        <div style=\"background: #ffffff; color: #000000; padding: 1rem 1.5rem; border-radius: 12px; margin-top: 1.5rem; text-align: center; font-weight: 600; border: 1px solid #e5e7eb;\">
            Labels ready! Switch to the <strong>Taste Prediction</strong> tab to generate insights
        </div>
        """, unsafe_allow_html=True)


# -----------------------
# Tab: Taste Prediction
# -----------------------
with tab_predict:
    st.markdown("""
    <div class="card" style="background: #ffffff; border: 1px solid #e5e7eb;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            
            <div>
                <h3 style="margin: 0; color: #000000; font-weight: 700;">Audience Analysis</h3>
                <p style="color: #000000; margin: 0.25rem 0 0 0;">Discover how different viewer segments will respond to your content</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    labels = st.session_state.get('labels')
    if not labels:
        st.markdown("""
        <div style="background: #ffffff; color: #000000; padding: 1.5rem; border-radius: 16px; text-align: center; border: 1px solid #e5e7eb;">
            <strong style="font-size: 1.1rem;">No Content Labeled Yet</strong>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">Please go to the "Label Script" tab first to analyze your content</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Initialize predictor once
        if 'predictor' not in st.session_state:
            st.session_state['predictor'] = get_predictor()

        pred_input = metadata_to_predictor_input(labels)
        
        # Display current content info with enhanced styling
        st.markdown("""
        <div class="card" style="background: #ffffff; border: 1px solid #e5e7eb;">
            <h4 style="margin-top: 0; color: #000000; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                Content Analysis Summary
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Title", labels.get('FRANCHISE_TITLE', 'Untitled'))
        with col2:
            st.metric("Genre", labels.get('Genre', 'Drama'))
        with col3:
            st.metric("Type", labels.get('CONTENT_TYPE', 'TV Show'))
        
        if st.button("Generate Predictions", type="primary", use_container_width=True):
            with st.spinner("Analyzing audience segments..."):
                predictor = st.session_state.get('predictor')
                # Inform if we're using synthetic predictor
                if getattr(predictor, 'is_synthetic', False):
                    st.info("Using synthetic predictions (TensorFlow model not loaded).")
                preds = predictor.predict_engagement(pred_input)
                st.session_state['predictions'] = preds
                st.balloons()
                st.success("Predictions generated successfully!")

        preds = st.session_state.get('predictions')
        if preds:
            st.markdown("""
            <div class="card" style="background: #ffffff; border: 1px solid #e5e7eb;">
                <h4 style="margin-top: 0; color: #000000; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📈</span> Prediction Results
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            colL, colR = st.columns([2, 1])
            with colL:
                st.markdown("""
            <h4 style="color: #000000; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                <span>🔥</span> Cluster Adoption Heatmap
            </h4>
            """, unsafe_allow_html=True)
                fig = build_adoption_heatmap(preds)
                _tmpl = "simple_white"
                fig.update_layout(template=_tmpl, font=dict(family="Inter, system-ui, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)

            with colR:
                st.markdown("#### Key Insights")
                # Headline insights
                top_adopt = max(preds.items(), key=lambda x: x[1]['adoption_rate'])
                st.metric(label="Highest Adoption", value=f"{top_adopt[1]['cluster_name']}", delta=f"{top_adopt[1]['adoption_rate']*100:.1f}%")
                total_viewers = sum(p['predicted_viewers'] for p in preds.values())
                st.metric(label="Total Viewers", value=f"{total_viewers:,}")
                
                # Calculate average engagement
                avg_engagement = sum(p['engagement_score'] for p in preds.values()) / len(preds)
                st.metric(label="Avg Engagement", value=f"{avg_engagement*100:.1f}%")

            # Additional bar chart for predicted viewers per cluster
            st.markdown("""
            <h4 style="color: #000000; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem;">
                Predicted Viewership Distribution
            </h4>
            """, unsafe_allow_html=True)
            _bar = pd.DataFrame([
                {
                    "Cluster": p["cluster_name"],
                    "Predicted Viewers": p["predicted_viewers"],
                    "Adoption": p["adoption_rate"],
                }
                for _, p in sorted(preds.items(), key=lambda kv: kv[1]["predicted_viewers"], reverse=True)
            ])
            try:
                import plotly.graph_objects as _go
                bar = _go.Figure(_go.Bar(
                    x=_bar["Cluster"],
                    y=_bar["Predicted Viewers"],
                    marker=dict(
                        color=_bar["Adoption"],
                        colorscale=[[0.0, '#fee2e2'], [0.5, '#fef3c7'], [1.0, '#dcfce7']],
                    ),
                ))
                bar.update_layout(
                    template="simple_white",
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(family="Inter, system-ui, sans-serif"),
                    yaxis_title="Predicted Viewers",
                )
                st.plotly_chart(bar, use_container_width=True)
            except Exception:
                st.bar_chart(_bar.set_index("Cluster")["Predicted Viewers"], use_container_width=True)

            # Detailed table with enhanced styling
            st.markdown("""
            <h4 style="color: #000000; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem;">
                Detailed Cluster Performance Metrics
            </h4>
            """, unsafe_allow_html=True)
            table_rows = []
            for cid, p in preds.items():
                table_rows.append({
                    'Cluster': p['cluster_name'],
                    'Adoption %': f"{p['adoption_rate']*100:.1f}%",
                    'Completion %': f"{p['completion_rate']*100:.1f}%",
                    'Engagement %': f"{p['engagement_score']*100:.1f}%",
                    'Predicted Viewers': f"{p['predicted_viewers']:,}",
                    'Level': p['engagement_level']
                })
            
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True)


# Footer removed
