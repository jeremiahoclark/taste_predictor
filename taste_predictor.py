import streamlit as st
import pandas as pd
import numpy as np
import json
from textwrap import dedent
import os
from groq import Groq
from dotenv import load_dotenv
import joblib

# Handle optional ML dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    st.warning("⚠️ sentence-transformers not available. Some features may be limited.")
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# --- Hardcoded Cluster Labels ---
CLUSTER_LABELS = {
  "0": "romance_thriller_darkcomedy",
  "1": "animated_comedy_satire",
  "2": "reality_comedy",
  "3": "animated_superhero",
  "4": "reality_glam_conflict",
  "5": "black_romance",
  "6": "biopic_drama",
  "7": "madea_comedy",
  "8": "celebrity_doc_music_drama",
  "9": "legacy_crime_empire",
  "10": "legal_justice_truth",
  "11": "holiday_family"
}

# --- Load Models and Data (cached) ---
@st.cache_resource
def load_models():
    load_dotenv()
    
    # Load SentenceTransformer if available
    embedder = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"Could not load SentenceTransformer: {e}")
            embedder = None
    
    # Load the ML models and data
    try:
        trained_models = joblib.load('trained_models.pkl')
        centroids = joblib.load('centroids.pkl')
        cluster_summary_df = pd.read_csv('cluster_summary.csv')
    except FileNotFoundError:
        st.error("Could not find model files. Please make sure 'trained_models.pkl', 'centroids.pkl', and 'cluster_summary.csv' are in the root directory.")
        trained_models, centroids, cluster_summary_df = {}, {}, pd.DataFrame()

    return embedder, trained_models, centroids, cluster_summary_df

embedder, trained_models, centroids, cluster_summary_df = load_models()


# --- Functions from Notebook (adapted for Streamlit) ---

def _call_llm(prompt: str, groq_model: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=groq_model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

def llm_define_metadata_v2(script_text: str, content_type: str, groq_model: str) -> dict:
    # In a real app, you might have a more robust way to get examples
    sample_examples = []

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
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Error parsing core attributes JSON from LLM: {e}")
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
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Error parsing detailed attributes JSON from LLM: {e}")
        return None

    metadata = {**core_attrs, **detailed_attrs}
    metadata['embedding_text'] = (f"{metadata.get('FRANCHISE_TITLE','')} | Genre: {metadata.get('Genre','')} | Sub: {metadata.get('Subgenre','')} | "
                                  f"Tonal: {metadata.get('Tonal_Comps','')} | Tropes: {metadata.get('Shared_Tropes','')} | "
                                  f"Diff: {metadata.get('Differential','')} | Protagonist: {metadata.get('Protagonist_Demo','')}")
    return metadata

def predict_from_metadata(
    approved_metadata: dict,
    trained_models: dict,
    centroids: dict,
    embedder_model: object,
    cluster_summary_df: pd.DataFrame
):
    if embedder_model is None:
        st.error("Cannot make predictions without embedder model. Please install sentence-transformers and restart the app.")
        return pd.DataFrame()
    
    content_embedding = embedder_model.encode([approved_metadata['embedding_text']])[0]
    content_embedding = content_embedding / (np.linalg.norm(content_embedding) + 1e-12)

    cluster_scores = []
    path_b_model = trained_models.get('path_b_rfr')

    if not path_b_model:
        st.error("Error: Path B model ('path_b_rfr') not found in trained_models.")
        return pd.DataFrame()

    for cluster_id, centroid in sorted(centroids.items()):
        scores = {'cluster_id': int(cluster_id)}
        X_B = np.concatenate([content_embedding, centroid]).reshape(1, -1)
        scores['p_adopt'] = np.clip(path_b_model.predict(X_B)[0], 0.0, 1.0)
        cluster_scores.append(scores)

    scores_df = pd.DataFrame(cluster_scores)

    if not cluster_summary_df.empty:
        scores_df = scores_df.merge(cluster_summary_df[['cluster_id', 'users', 'cluster_name']], on='cluster_id', how='left')
        if 'users' in scores_df.columns:
            scores_df['expected_adopters'] = scores_df['p_adopt'] * scores_df['users']

    return scores_df.sort_values('p_adopt', ascending=False)


# --- Streamlit App ---

st.title("Taste Predictor")

# --- 1. Upload Script ---
uploaded_file = st.file_uploader("Upload a script file (.txt)", type="txt")
content_type = st.selectbox("Select Content Type", ["TV Show", "Feature Film"])
groq_model = st.text_input("Groq Model", "moonshotai/kimi-k2-instruct")


if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


if uploaded_file is not None and st.button("Generate Metadata"):
    script_text = uploaded_file.read().decode("utf-8")
    with st.spinner("Generating metadata..."):
        st.session_state.metadata = llm_define_metadata_v2(script_text, content_type, groq_model)
    st.session_state.predictions = None


# --- 2. Review and Edit Metadata ---
if st.session_state.metadata:
    st.subheader("Review and Edit Metadata")
    
    edited_metadata = {}
    for key, value in st.session_state.metadata.items():
        if isinstance(value, list):
            edited_metadata[key] = st.text_area(f"Edit {key}", ", ".join(map(str, value))).split(', ')
        else:
            edited_metadata[key] = st.text_area(f"Edit {key}", value)
    
    st.session_state.metadata = edited_metadata

    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            st.session_state.predictions = predict_from_metadata(
                st.session_state.metadata,
                trained_models,
                centroids,
                embedder,
                cluster_summary_df
            )

# --- 3. Visualize Output ---
if st.session_state.predictions is not None:
    st.subheader("Prediction Results")

    # Create a 3x4 grid
    cols = st.columns(4)
    
    # Sort predictions by p_adopt
    sorted_predictions = st.session_state.predictions.sort_values('p_adopt', ascending=False)

    for i, row in enumerate(sorted_predictions.itertuples()):
        cluster_id = str(row.cluster_id)
        label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        p_adopt = row.p_adopt
        
        with cols[i % 4]:
            st.metric(label=label, value=f"{p_adopt:.2%}")

    st.dataframe(st.session_state.predictions)