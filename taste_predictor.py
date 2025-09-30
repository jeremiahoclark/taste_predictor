import streamlit as st
import pandas as pd
import numpy as np
import json
from textwrap import dedent
import os
from groq import Groq
from dotenv import load_dotenv
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

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

# --- ROI Configuration ---
VALUE_PER_ADOPTER = 5.0
MONETIZATION_RATE = 0.01

# --- Database Connection ---
def get_db_connection():
    """Create PostgreSQL connection using environment variables"""
    return psycopg2.connect(
        dbname=os.environ.get('PGDATABASE', 'neondb'),
        user=os.environ.get('PGUSER', 'neondb_owner'),
        password=os.environ.get('PGPASSWORD', 'npg_31yBGbtAdqoK'),
        host=os.environ.get('PGHOST', 'ep-shy-pond-aeuphyma.c-2.us-east-2.aws.neon.tech'),
        port=os.environ.get('PGPORT', '5432'),
        sslmode='require'
    )

def load_user_clusters():
    """Load user cluster assignments from database"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT account_id, cluster_id FROM user_cluster_assignments", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading user clusters: {e}")
        return pd.DataFrame()

def load_cluster_summary():
    """Get user counts per cluster"""
    try:
        user_clusters = load_user_clusters()
        if user_clusters.empty:
            return pd.DataFrame()
        return user_clusters.groupby('cluster_id').size().reset_index(name='users')
    except Exception as e:
        st.error(f"Error loading cluster summary: {e}")
        return pd.DataFrame()

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
            scores_df = scores_df.drop(columns=['users'])

    return scores_df.sort_values('p_adopt', ascending=False)

def compute_roi_value(pred_df: pd.DataFrame, user_counts: pd.DataFrame, avg_completion: float = 0.5) -> float:
    """Compute predicted value from adoption predictions."""
    df = pred_df.merge(user_counts, on='cluster_id')
    expected_adopters = (df['p_adopt'] * df['users']).sum()
    value = expected_adopters * VALUE_PER_ADOPTER * avg_completion
    return value

def estimate_cost_from_metadata(metadata: dict) -> float:
    """Estimate cost based on genre/subgenre from database."""
    try:
        conn = get_db_connection()
        genre = metadata.get('Genre', '')
        subgenre = metadata.get('Subgenre', '')

        query = """
            SELECT AVG(l.total_cost) as avg_cost,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.total_cost) as median_cost
            FROM licenses l
            JOIN franchise_metadata fm ON l.title_id = fm.franchise_id
            WHERE fm.genre = %s AND fm.subgenre = %s
        """

        result = pd.read_sql_query(query, conn, params=(genre, subgenre))
        conn.close()

        if not result.empty and result['median_cost'].iloc[0] is not None:
            return float(result['median_cost'].iloc[0])
        return 90000.0  # Fallback cost
    except Exception as e:
        st.warning(f"Could not estimate cost from database: {e}. Using default.")
        return 90000.0

def predict_roi(metadata: dict, budget: float, trained_models: dict, centroids: dict, embedder_model: object) -> dict:
    """Predict ROI for a script given budget."""
    # Get predictions
    user_counts = load_cluster_summary()
    if user_counts.empty:
        st.error("Could not load user data for ROI calculation")
        return None

    pred_df = predict_from_metadata(metadata, trained_models, centroids, embedder_model, user_counts)

    # Calculate value and cost
    estimated_value = compute_roi_value(pred_df, user_counts)
    estimated_cost = estimate_cost_from_metadata(metadata)

    # Calculate ROI
    net_profit = estimated_value - estimated_cost
    roi_percentage = (net_profit / estimated_cost) * 100 if estimated_cost > 0 else 0

    # Determine if within budget
    within_budget = estimated_cost <= budget

    # Calculate expected adopters
    df_with_users = pred_df.merge(user_counts, on='cluster_id')
    expected_adopters = (df_with_users['p_adopt'] * df_with_users['users']).sum()

    return {
        'estimated_cost': estimated_cost,
        'estimated_value': estimated_value,
        'net_profit': net_profit,
        'roi_percentage': roi_percentage,
        'within_budget': within_budget,
        'expected_adopters': expected_adopters,
        'pred_df': pred_df
    }


# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }

    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #cccccc;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Step indicators */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        gap: 1rem;
    }

    .step-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border-color: #333;
    }

    .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #1a1a1a 0%, #3d3d3d 100%);
        color: white;
        border-radius: 50%;
        line-height: 40px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .step-title {
        font-weight: 600;
        color: #1a1a1a;
        margin: 0.5rem 0;
        font-size: 1rem;
    }

    .step-desc {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #d0d0d0;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #333;
        background: #fafafa;
    }

    /* Metadata cards */
    .metadata-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1a1a1a;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Recommendation banner */
    .recommendation-banner {
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }

    .recommendation-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shine 3s infinite;
    }

    @keyframes shine {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }

    .recommendation-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #f8f8f8 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1a1a1a;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    /* Cluster cards */
    .cluster-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }

    .cluster-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #333;
    }

    .cluster-card-top {
        background: linear-gradient(135deg, #1a1a1a 0%, #3d3d3d 100%);
        color: white;
        border: 2px solid #1a1a1a;
    }

    .cluster-card-top:hover {
        transform: scale(1.08);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        background: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    div[role="radiogroup"] label:hover {
        border-color: #333;
        background: #f5f5f5;
    }

    /* Text areas */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #333;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #1a1a1a 0%, #3d3d3d 100%);
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit App ---

# Modern header
st.markdown("""
<div class="main-header">
    <h1>Content Engagement Predictor</h1>
    <p>Audience engagement analysis for scripts and pitches</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Engagement Predictor", "ROI Predictor"])

# Initialize session state
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'roi_results' not in st.session_state:
    st.session_state.roi_results = None

groq_model = "moonshotai/kimi-k2-instruct-0905"  # Hidden from UI

# ==================== TAB 1: ENGAGEMENT PREDICTOR ====================
with tab1:
    # Step indicators
    st.markdown("""
    <div class="step-container">
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-title">Upload Script</div>
            <div class="step-desc">Text document or pitch</div>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-title">Review Metadata</div>
            <div class="step-desc">Edit if needed</div>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-title">View Engagement</div>
            <div class="step-desc">Audience predictions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 1. Upload Script ---
    uploaded_file = st.file_uploader("Upload a script file (.txt)", type="txt", key="engagement_upload")
    content_type = st.radio("Select Content Type", ["TV Show", "Feature Film"], horizontal=True)

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
                edited_metadata[key] = st.text_area(f"Edit {key}", ", ".join(map(str, value)), key=f"engagement_{key}").split(', ')
            else:
                edited_metadata[key] = st.text_area(f"Edit {key}", value, key=f"engagement_{key}")

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
        # Sort predictions by p_adopt
        sorted_predictions = st.session_state.predictions.sort_values('p_adopt', ascending=False)

        # Calculate engagement index (average of top 3 clusters)
        top_3_scores = sorted_predictions.head(3)['p_adopt'].values
        engagement_index = top_3_scores.mean()

        # Determine recommendation level
        if engagement_index > 0.65:
            recommendation = "Strong Opportunity"
            color = "green"
        elif engagement_index >= 0.45:
            recommendation = "Moderate Opportunity"
            color = "orange"
        else:
            recommendation = "Weak Opportunity"
            color = "red"

        # Generate qualitative insight
        top_cluster_name = CLUSTER_LABELS.get(str(sorted_predictions.iloc[0]['cluster_id']), "Unknown")
        top_score = sorted_predictions.iloc[0]['p_adopt']
        second_score = sorted_predictions.iloc[1]['p_adopt'] if len(sorted_predictions) > 1 else 0
        third_score = sorted_predictions.iloc[2]['p_adopt'] if len(sorted_predictions) > 2 else 0

        top_3_names = [CLUSTER_LABELS.get(str(sorted_predictions.iloc[i]['cluster_id']), "Unknown") for i in range(min(3, len(sorted_predictions)))]

        if top_score - second_score > 0.15:
            insight = f"Strong appeal to **{top_cluster_name}** audience"
        elif top_score < 0.60:
            insight = f"Niche content with focused appeal to **{top_cluster_name}** viewers"
        else:
            insight = f"Broad engagement across **{top_3_names[0]}**, **{top_3_names[1]}**, and **{top_3_names[2]}** audiences"

        # Display Recommendation Section
        st.markdown("### Recommendation")

        # Overall recommendation banner
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">{recommendation}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Key insight
        st.markdown(f"**Key Insight:** {insight}")

        # Engagement score
        st.metric("Engagement Score", f"{engagement_index:.0%}", help="Average engagement probability of top 3 audience clusters")

        st.markdown("---")

        # Detailed breakdown
        st.subheader("Audience Engagement Breakdown")

        # Create a 3x4 grid with better visual hierarchy
        cols = st.columns(4)

        for i, row in enumerate(sorted_predictions.itertuples()):
            cluster_id = str(row.cluster_id)
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            p_adopt = row.p_adopt

            # Highlight top 3 clusters
            is_top_3 = i < 3

            with cols[i % 4]:
                if is_top_3:
                    st.markdown(f"<div style='margin-bottom: 10px;'><strong>TOP: {label}</strong></div>", unsafe_allow_html=True)
                st.metric(label=label if not is_top_3 else "", value=f"{p_adopt:.0%}")

        # Optional: Show detailed table in expander
        with st.expander("View Detailed Data"):
            st.dataframe(st.session_state.predictions, use_container_width=True)

# ==================== TAB 2: ROI PREDICTOR ====================
with tab2:
    st.markdown("""
    <div class="step-container">
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-title">Upload Script</div>
            <div class="step-desc">Text document or pitch</div>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-title">Set Budget</div>
            <div class="step-desc">Enter acquisition budget</div>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-title">View ROI</div>
            <div class="step-desc">Financial projections</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 1. Upload Script ---
    roi_uploaded_file = st.file_uploader("Upload a script file (.txt)", type="txt", key="roi_upload")
    roi_content_type = st.radio("Select Content Type", ["TV Show", "Feature Film"], horizontal=True, key="roi_content_type")

    # --- 2. Budget Input ---
    budget = st.number_input("Enter Budget ($)", min_value=0, value=100000, step=10000)

    if roi_uploaded_file is not None and st.button("Analyze ROI", key="roi_analyze"):
        script_text = roi_uploaded_file.read().decode("utf-8")
        with st.spinner("Analyzing script and calculating ROI..."):
            # Generate metadata
            roi_metadata = llm_define_metadata_v2(script_text, roi_content_type, groq_model)
            if roi_metadata:
                # Calculate ROI
                st.session_state.roi_results = predict_roi(
                    roi_metadata,
                    budget,
                    trained_models,
                    centroids,
                    embedder
                )
            else:
                st.error("Failed to generate metadata from script")

    # --- 3. Display ROI Results ---
    if st.session_state.roi_results is not None:
        results = st.session_state.roi_results

        # Determine recommendation
        if results['roi_percentage'] > 50:
            roi_recommendation = "Strong Investment"
            roi_color = "green"
        elif results['roi_percentage'] > 0:
            roi_recommendation = "Moderate Investment"
            roi_color = "orange"
        else:
            roi_recommendation = "Weak Investment"
            roi_color = "red"

        st.markdown("### ROI Analysis")

        # ROI Banner
        st.markdown(f"""
        <div style="background-color: {roi_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">{roi_recommendation}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Cost", f"${results['estimated_cost']:,.0f}")
        with col2:
            st.metric("Estimated Value", f"${results['estimated_value']:,.0f}")
        with col3:
            st.metric("ROI", f"{results['roi_percentage']:.1f}%")

        st.markdown("---")

        # Additional Details
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Net Profit", f"${results['net_profit']:,.0f}")
        with col5:
            st.metric("Expected Adopters", f"{results['expected_adopters']:,.0f}")

        # Budget Status
        if results['within_budget']:
            st.success(f"✓ This acquisition is within your ${budget:,.0f} budget")
        else:
            st.warning(f"⚠ This acquisition exceeds your ${budget:,.0f} budget by ${results['estimated_cost'] - budget:,.0f}")

        st.markdown("---")

        # Cluster Breakdown
        st.subheader("Value by Audience Cluster")

        user_counts = load_cluster_summary()
        if not user_counts.empty:
            pred_with_users = results['pred_df'].merge(user_counts, on='cluster_id')
            pred_with_users['expected_adopters'] = pred_with_users['p_adopt'] * pred_with_users['users']
            pred_with_users['cluster_value'] = pred_with_users['expected_adopters'] * VALUE_PER_ADOPTER * 0.5
            pred_with_users['cluster_name'] = pred_with_users['cluster_id'].astype(str).map(CLUSTER_LABELS)
            pred_with_users = pred_with_users.sort_values('cluster_value', ascending=False)

            # Display top value clusters
            cols = st.columns(4)
            for i, row in enumerate(pred_with_users.head(12).itertuples()):
                with cols[i % 4]:
                    st.metric(
                        label=row.cluster_name if hasattr(row, 'cluster_name') else f"Cluster {row.cluster_id}",
                        value=f"${row.cluster_value:,.0f}",
                        help=f"P(adopt): {row.p_adopt:.1%}, Users: {row.users:,}"
                    )

        # Detailed table
        with st.expander("View Detailed Breakdown"):
            if not user_counts.empty:
                st.dataframe(pred_with_users[['cluster_name', 'p_adopt', 'users', 'expected_adopters', 'cluster_value']], use_container_width=True)