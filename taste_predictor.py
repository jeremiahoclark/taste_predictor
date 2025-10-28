import streamlit as st
import pandas as pd
import numpy as np
import json
from textwrap import dedent
import os
import requests
from dotenv import load_dotenv
import joblib
from urllib.parse import urlparse
import urllib.request
import ssl
from typing import List, Dict, Any, Optional

# Handle optional ML dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    st.warning("⚠️ sentence-transformers not available. Some features may be limited.")
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# Handle PDF dependencies
try:
    from pypdf import PdfReader
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    PdfReader = None

# Handle OCR dependencies (optional for image-based PDFs)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    HAS_OCR_SUPPORT = True
except ImportError:
    HAS_OCR_SUPPORT = False

# Handle database dependencies
try:
    import psycopg2
    HAS_DB_SUPPORT = True
except ImportError:
    HAS_DB_SUPPORT = False

# --- Hardcoded Cluster Labels ---
CLUSTER_LABELS = {
  "0": "Romance-Infused, Suspense-Driven Stories",
  "1": "Celebrity-and Culture-Driven Comedy",
  "2": "Reality-Driven, Socially Chaotic Comedy",
  "3": "Hidden Identity, Crime, and Supernatural Thrills",
  "4": "Adult Relationship and Life-Stage Stories",
  "5": "Community, Music, and Relationship Stories",
  "6": "Ambition, Fame, and Erotic Thrillers",
  "7": "Faith-Tinged, Family-Focused Comedy",
  "8": "Star-and Legacy-Centered Documentary",
  "9": "Crime and Dynasty Family Sagas",
  "10": "Legal Battles and Female Bonds",
  "11": "Holiday Romance and Wish Fulfillment"
}

# --- Detailed Cluster Information ---
CLUSTER_INFO = {
    0: {
        "name": "Romance-Infused, Suspense-Driven Stories",
        "subtitle": "Love, Lies, and Power Plays",
        "description": "Stories where love and desire intersect with secrets, danger, and high-stakes intrigue. From hidden royalty and obsessive relationships to family betrayals and psychological threats, these narratives blend romance, attraction, and even obsession with suspenseful drama. Viewers drawn here enjoy emotionally charged plots where passion and peril collide.",
        "genres": ["Thriller", "Romance", "Dark Comedy"],
        "subgenres": ["Holiday Romance", "Mystery", "Slasher"],
        "tropes": ["secret royalty", "cultural clash", "obsessive love", "isolation of victim", "technology as weapon"]
    },
    1: {
        "name": "Celebrity- and Culture-Driven Comedy",
        "subtitle": "Satire, Fame, and Hustle",
        "description": "Comedies, satires, and biographical dramas that explore celebrity life, music history, and urban culture. From animated coming-of-age stories and stand-up showcases to music biopics and satirical Hollywood send-ups, these narratives mix humor with cultural commentary. Viewers drawn here enjoy both the laughs and the behind-the-scenes look at fame, fortune, and the struggles that come with them. Audiences are drawn to the way these series lets them get an inside look at a life of luxury while showing the drawbacks that come with it. A critique and celebration at the same time.",
        "genres": ["Comedy", "Adult Animation", "Biographical Drama"],
        "subgenres": ["Comedy", "Satire", "Music Biography"],
        "tropes": ["coming of age", "betrayal", "riches-to-rags/rags-to-riches", "rise to fame", "downfall"]
    },
    2: {
        "name": "Reality-Driven, Socially Chaotic Comedy",
        "subtitle": "Everyday Life and Misadventures",
        "description": "Reality shows, docuseries, and comedic urban stories where family, friends, and social dynamics collide. From prank dating shows and hidden-camera reveals to influencer feuds and dysfunctional gatherings, these narratives mix humor with relatable chaos and unexpected misadventures. Viewers enjoy laugh-out-loud moments rooted in real-life awkwardness, social drama, everyday mishaps, and resilient characters navigating life's ups and downs.",
        "genres": ["Reality-TV", "Reality", "Comedy"],
        "subgenres": ["Prank Dating Show", "Docuseries", "Family Sitcom"],
        "tropes": ["hidden camera reveals", "overcoming adversity", "culture clash", "cringe comedy", "hijinks and antics"]
    },
    3: {
        "name": "Hidden Identity, Crime, and Supernatural Thrills",
        "subtitle": "Secrets, Power, and Betrayal",
        "description": "A mix of animated odd-couple comedies, teen superhero adventures, and crime-driven sagas that explore hidden identities and high-stakes deception. Stories range from lighthearted animation with quirky dynamics to dark dramas about family empires, infidelity, and betrayal. They often employ melodrama with a genre-edge that helps elevate the experience for more niche audiences. Viewers here gravitate toward narratives where living a double life—whether as a ghost-powered teen or a crime boss—creates both comedy and danger.",
        "genres": ["Animation", "Superhero", "Crime Drama"],
        "subgenres": ["Comedy", "Teen Superhero", "Family Saga"],
        "tropes": ["odd couple dynamic", "double life", "interracial affairs", "couples therapy gone wrong", "infidelity revenge"]
    },
    4: {
        "name": "Adult Relationship and Life-Stage Stories",
        "subtitle": "Love, Friendship, and Second Chances",
        "description": "Comedies, dramas, and docuseries exploring love, relationships, and personal growth in adulthood. From divorce recovery and midlife crises to dating disasters and redemption arcs, these stories balance humor with heartfelt lessons. Viewers drawn here enjoy character-driven narratives about love, friendship, and second chances. They focus either on the lives of the wealthy or trendy (for audiences to live vicariously) or on more grounded life events like financial or marital issues (to give audiences a sense of relatability or kinship).",
        "genres": ["Comedy", "Comedy Drama", "Reality"],
        "subgenres": ["Drama", "Romantic Drama", "Docuseries"],
        "tropes": ["redemption arc", "divorce recovery", "relationship drama", "dating app gone wrong", "midlife crisis"]
    },
    5: {
        "name": "Community, Music, and Relationship Stories",
        "subtitle": "Love, Celebration, and Homecoming",
        "description": "Dramas, comedies, and biographical stories that explore love, music, and the pull of community. These narratives often center on characters navigating romance, personal hardship, or artistic ambition, while finding strength in family bonds or cultural celebration. Whether through sugar relationships, hometown reunions, or rise-to-fame music stories, this cluster appeals to those drawn to heartfelt journeys with both joy and struggle. Cultural traditions (community, music, gatherings, etc) often serve as a reflection of the drama/themes at play in the material, giving a stronger catharsis than narrative alone can achieve.",
        "genres": ["Drama", "Biographical Drama"],
        "subgenres": ["Romance", "Comedy", "Music Biography"],
        "tropes": ["sugar daddy relationships", "solo careers vs. group success", "community celebration", "family bonds", "returning home"]
    },
    6: {
        "name": "Ambition, Fame, and Erotic Thrillers",
        "subtitle": "Power, Desire, and Betrayal",
        "description": "Crime dramas, biographical stories, and relationship-centered sitcoms where ambition and desire drive the action. From maintaining criminal empires to chasing stardom in music or navigating messy romances at work, these narratives mix glitz, drama, and interpersonal conflict. Viewers here enjoy tales of striving for power, fame, and love—often at the cost of loyalty and stability. Protagonists often lose sympathy and either regain it or dive deeper into their moral failings.",
        "genres": ["Biographical Drama", "Crime Drama", "Sitcom"],
        "subgenres": ["Music Biography", "Erotic Thriller", "Personality/Celeb Sitcom"],
        "tropes": ["rise to fame", "workplace antics", "romantic entanglements", "family dynamics", "workplace drama"]
    },
    7: {
        "name": "Faith-Tinged, Family-Focused Comedy",
        "subtitle": "Faith, Family, and Festive Fiascos",
        "description": "Where faith meets family dysfunction and holiday hijinks. These stories balance laugh-out-loud slapstick comedy with heartfelt lessons, often set against church pews, Christmas trees, or family gatherings that go off the rails. Humor often comes from the clash between lionized/solemn traditions and dysfunctional human nature. They often play off of generational differences and community tropes. They usually have a distinctive, comic (and at times, antagonistic) figure to represent the older generation to clash with the younger generation (audience insert characters). There are elements of melodrama with an overall sense that family bonds rise above all adversity/conflict.",
        "genres": ["Comedy", "Reality", "Holiday"],
        "subgenres": ["Crime Comedy", "Docu-drama", "Holiday Comedy"],
        "tropes": ["cross", "pastor's kids rebelling", "dressing protagonist", "dance sequence", "girlfriend as antagonist"]
    },
    8: {
        "name": "Star- and Legacy-Centered Documentary",
        "subtitle": "Behind the Music and the Moments",
        "description": "Documentaries, reality shows, and biopics offering an intimate look at celebrities, musicians, and cultural figures. These stories reveal the triumphs, struggles, and secrets behind public personas—whether through milestone celebrations, behind-the-scenes concert films, or family and church politics. Viewers drawn here seek inspiration, empowerment, and the drama of real lives unfolding on and off stage. It often requires a buy-in with the central figure. Whether it be an appeal to fans or audiences with a passing interest, they thrive off and foster a healthy parasocial bond with the protagonist.",
        "genres": ["Documentary", "Reality", "Biographical Drama"],
        "subgenres": ["Travel Documentary", "Celebrity Reality", "Concert Film"],
        "tropes": ["behind-the-scenes", "church politics", "family secrets", "redemption arcs"]
    },
    9: {
        "name": "Crime and Dynasty Family Sagas",
        "subtitle": "Empires, Loyalty, and Betrayal",
        "description": "Crime sagas, western thrillers, and biographical dramas centered on family empires, hidden identities, and power struggles. These narratives often pit family loyalty against personal ambition, explore old money versus new money, and spotlight female empowerment within dynasties and industries. Rather than focus on one singular character, they are more focused on the ensemble at play. Viewers here enjoy sprawling stories where wealth, crime, and reputation come with high emotional and moral stakes.",
        "genres": ["Crime Drama", "Western", "Biographical Drama"],
        "subgenres": ["Family Saga", "Thriller", "Music Biopic"],
        "tropes": ["family loyalty vs personal ambition", "female empowerment", "family dynasties", "old money vs new money", "hidden identities"]
    },
    10: {
        "name": "Legal Battles and Female Bonds",
        "subtitle": "Justice, Friendship, and Defiance",
        "description": "Legal dramas, comedy-dramas, and biographical stories centered on the pursuit of justice and the bonds of sisterhood. These narratives highlight women navigating divorce, friendship, wrongful convictions, and David vs. Goliath legal battles. Whether explicitly or not, these series play off of the feeling of a 'rigged system' and find catharsis in the ways the protagonists overcome adversity. Viewers drawn here connect with empowering tales of resilience, whether in the courtroom, the music industry, or the struggles of personal relationships.",
        "genres": ["Legal Drama", "Comedy-Drama", "Biographical Drama"],
        "subgenres": ["Crime Drama", "Relationship Drama", "Legal Drama"],
        "tropes": ["female friendship", "divorce", "artist-label relationships", "federal investigation", "personal transformation"]
    },
    11: {
        "name": "Holiday Romance and Wish Fulfillment",
        "subtitle": "Magic, Love, and Second Chances",
        "description": "Romantic comedies, holiday films, and dramas steeped in wish fulfillment, festive magic, and second chances. From small-town Christmas romances to sugar relationship dramas, these stories deliver warmth, love, and emotional renewal. Viewers drawn here enjoy escapist tales where love is tested, rekindled, or discovered during the most magical of seasons. Less interested in satire or comedy and more on a pleasant, easy watching vibe.",
        "genres": ["Comedy", "Holiday", "Drama"],
        "subgenres": ["Romantic Comedy", "Romance", "Drama"],
        "tropes": ["wish fulfillment", "holiday magic", "sugar daddy relationships", "female friendship", "small town Christmas"]
    }
}

# --- ROI Configuration ---
VALUE_PER_ADOPTER = 5.0
MONETIZATION_RATE = 0.01
AVERAGE_COMPLETION_RATE = 0.5

# --- Database Connection (using Replit's native DB or fallback to mock data) ---
@st.cache_data(ttl=600)
def load_user_clusters():
    """Load user cluster assignments - try Replit DB first, fallback to mock data"""
    try:
        # Try to use Replit's native database if available
        from replit import db as replit_db

        # Check if we have cluster data in Replit DB
        if 'user_clusters' in replit_db.keys():
            data = json.loads(replit_db['user_clusters'])
            return pd.DataFrame(data)
    except:
        pass

    # Fallback: Generate mock data based on cluster distribution
    # This simulates realistic user distribution across clusters
    np.random.seed(42)
    cluster_sizes = {
        0: 1500, 1: 1200, 2: 1800, 3: 900,
        4: 1100, 5: 800, 6: 1300, 7: 600,
        8: 1000, 9: 700, 10: 950, 11: 850
    }

    data = []
    account_id = 1
    for cluster_id, size in cluster_sizes.items():
        for _ in range(size):
            data.append({'account_id': account_id, 'cluster_id': cluster_id})
            account_id += 1

    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def load_cluster_summary(target_total: int = 3_000_000):
    """Get user counts per cluster, extrapolated to target total users"""
    try:
        user_clusters = load_user_clusters()
        if user_clusters.empty:
            return pd.DataFrame()
        df = user_clusters.groupby('cluster_id').size().reset_index(name='users')

        # Calculate current total and extrapolate to target users
        current_total = df['users'].sum()
        scaling_factor = target_total / current_total

        # Scale up proportionally
        df['users'] = (df['users'] * scaling_factor).round().astype(int)

        # Adjust to ensure exact target total (handle rounding errors)
        actual_total = df['users'].sum()
        if actual_total != target_total:
            diff = target_total - actual_total
            # Add/subtract difference to largest cluster
            largest_cluster_idx = df['users'].idxmax()
            df.loc[largest_cluster_idx, 'users'] += diff

        df['cluster_name'] = df['cluster_id'].astype(str).map(CLUSTER_LABELS)
        return df
    except Exception as e:
        st.error(f"Error loading cluster summary: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_taste_vocabulary():
    """Fetch all unique taste tags from the database to guide LLM metadata generation"""
    if not HAS_DB_SUPPORT:
        return ""

    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return ""

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        # Get all taste descriptions
        cur.execute('SELECT taste_descriptions FROM taste_descriptions')
        rows = cur.fetchall()

        # Extract all unique tags
        all_tags = set()
        for (desc,) in rows:
            if desc:
                tags = [tag.strip() for tag in desc.split(',')]
                all_tags.update(tags)

        cur.close()
        conn.close()

        # Sort and format for prompt
        sorted_tags = sorted(all_tags)
        return ", ".join(sorted_tags)
    except Exception as e:
        st.warning(f"Could not load taste vocabulary from database: {e}")
        return ""

# Load environment variables (works with both .env files and Replit secrets)
load_dotenv()

# --- Load Models and Data (cached) ---
@st.cache_resource
def load_models():
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

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using pypdf, fallback to OCR if available and needed."""
    if not HAS_PDF_SUPPORT:
        st.error("PDF support not available. Please install pypdf.")
        return None

    try:
        # First try pypdf for text extraction
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer

        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        # If text extraction yielded very little text, try OCR if available
        if len(text.strip()) < 100:
            if HAS_OCR_SUPPORT:
                st.info("PDF appears to be image-based. Running OCR (this may take a moment)...")
                images = convert_from_bytes(pdf_bytes)
                text = ""
                for i, image in enumerate(images):
                    text += pytesseract.image_to_string(image) + "\n"
            else:
                st.warning("PDF appears to be image-based, but OCR support is not available. Showing extracted text (may be incomplete).")

        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def _call_llm(prompt: str, model: str) -> str:
    """Call OpenRouter API with the specified model"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OPENROUTER_API_KEY not found in environment variables")
        return None

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        return None

def llm_define_metadata_v2(script_text: str, content_type: str, llm_model: str, retry_feedback: str = None) -> dict:
    # In a real app, you might have a more robust way to get examples
    sample_examples = []

    # Get taste vocabulary from database
    taste_vocab = get_taste_vocabulary()

    # Add retry feedback if provided
    retry_instruction = ""
    if retry_feedback:
        retry_instruction = f"""

        IMPORTANT USER FEEDBACK FROM PREVIOUS ATTEMPT:
        The user indicated the previous analysis was incorrect or incomplete. Please pay special attention to:
        {retry_feedback}
        """

    summary_prompt = dedent(f"""
        You are a professional script reader for a major studio.
        Your task is to summarize the following script in 3-4 concise sentences.
        Focus on the main plot, primary characters, setting, and overall tone.
        Do not add any preamble or explanation. Provide only the summary.
        {retry_instruction}

        SCRIPT TEXT:
        ---
        {script_text[:2000]}
        ---
    """)
    summary = _call_llm(summary_prompt, llm_model)
    if not summary:
        return None

    # Build vocabulary guidance for the prompt
    vocab_guidance = ""
    if taste_vocab:
        vocab_guidance = f"""

        TASTE VOCABULARY REFERENCE:
        When describing Genre and Subgenre, try to reuse existing genre categories and descriptive tags from our content library when possible. Here are the taste descriptors we use:
        {taste_vocab}

        Use these tags to inform your Genre and Subgenre choices. For Genre, use primary categories like "Romance", "Thriller", "Drama", "Comedy", "Horror", "Action", etc. For Subgenre, combine relevant tags that capture the specific flavor (e.g., "Romantic comedy", "Black family drama", "Holiday romance", "Crime thriller", etc.).
        """

    core_fields = ['FRANCHISE_TITLE', 'logline', 'Genre', 'Subgenre', 'CONTENT_TYPE']
    core_prompt = dedent(f"""
        You are a professional script analyst and development executive. Your task is to read a script summary and extract its core creative DNA.

        TASK:
        Analyze the provided script summary and identify its core attributes.
        {vocab_guidance}

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
        core_attrs_str = _call_llm(core_prompt, llm_model)
        if core_attrs_str is None:
            return None
        core_attrs = json.loads(core_attrs_str)
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Error parsing core attributes JSON from LLM: {e}")
        return None

    detailed_fields = ['Tonal_Comps', 'Shared_Tropes', 'Differential', 'Protagonist_Demo']

    # Add vocabulary guidance for detailed attributes
    detailed_vocab_guidance = ""
    if taste_vocab:
        detailed_vocab_guidance = f"""

        TASTE VOCABULARY REFERENCE:
        When identifying Shared_Tropes, use descriptive tags from our content library vocabulary when possible. Here are examples of tropes and themes we recognize:
        {taste_vocab}

        These tags represent common themes, settings, character types, tones, and narrative elements. Use them to describe the story's tropes in familiar terms.
        """

    detailed_prompt = dedent(f"""
        You are a professional script analyst and development executive. Your task is to read a script summary and its core attributes, then extract detailed descriptive elements.

        TASK:
        Analyze the provided summary and core attributes to identify deeper creative and demographic details.
        - For 'Tonal_Comps', list 3-5 existing movies or TV shows with a similar tone and feel.
        - For 'Shared_Tropes', identify 3-5 common narrative tropes or themes present in the story. Use concise, descriptive tags.
        - For 'Differential', explain in one sentence what makes this concept unique or fresh.
        - For 'Protagonist_Demo', describe the main character(s) including age, gender, and profession if known.
        {detailed_vocab_guidance}

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
        detailed_attrs_str = _call_llm(detailed_prompt, llm_model)
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

def llm_validate_predictions(
    predictions_df: pd.DataFrame,
    metadata: dict,
    llm_model: str
) -> pd.DataFrame:
    """
    Use LLM to validate and adjust model predictions based on content understanding
    and BET+ audience fit.
    """
    # Prepare cluster predictions for LLM
    cluster_predictions = []
    for _, row in predictions_df.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_name = CLUSTER_LABELS.get(str(cluster_id), f"Cluster {cluster_id}")
        cluster_predictions.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'model_prediction': float(row['p_adopt'])
        })

    validation_prompt = dedent(f"""
        You are an expert content strategist for BET+, a streaming platform focused on Black entertainment and culture.

        PLATFORM CONTEXT:
        BET+ serves a primarily Black American audience and features content including:
        - Black stories, perspectives, and talent
        - Urban contemporary entertainment
        - Faith-based and family content
        - R&B, hip-hop, and Black music culture
        - Black romance, drama, and comedy
        - True crime and documentaries featuring Black communities
        - Legacy content from BET network

        CONTENT TO EVALUATE:
        Title: {metadata.get('FRANCHISE_TITLE', 'Unknown')}
        Genre: {metadata.get('Genre', 'Unknown')}
        Subgenre: {metadata.get('Subgenre', 'Unknown')}
        Tonal Comps: {metadata.get('Tonal_Comps', 'None provided')}
        Tropes: {metadata.get('Shared_Tropes', 'None provided')}
        Differential: {metadata.get('Differential', 'None provided')}
        Protagonist: {metadata.get('Protagonist_Demo', 'Unknown')}

        MODEL PREDICTIONS:
        Our ML model predicted these engagement probabilities for each audience cluster:
        {json.dumps(cluster_predictions, indent=2)}

        TASK:
        The model has known issues with cluster overlap and may not accurately reflect BET+ audience preferences.
        Please review each prediction and suggest adjustments based on:
        1. Content relevance to Black culture and BET+ audience
        2. Star power and cultural recognition (e.g., Marvel, Tyler Perry, major Black stars)
        3. Genre appeal to specific clusters
        4. Realistic engagement expectations

        GUIDELINES:
        - High-profile Black content (Marvel with Black leads, Tyler Perry, etc.) should score 0.7-0.9 across multiple clusters
        - Abstract/niche content should score 0.3-0.5
        - Content misaligned with BET+ audience should score 0.1-0.3
        - Faith-based content should boost "Faith-Tinged, Family-Focused Comedy" scores
        - Reality/documentary should boost reality and documentary clusters
        - Romantic content should boost romance clusters
        - Crime/thriller should boost crime saga and thriller clusters

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with this structure (no other text):
        {{
            "adjustments": [
                {{"cluster_id": 0, "adjusted_prediction": 0.75}},
                {{"cluster_id": 1, "adjusted_prediction": 0.45}},
                ...
            ]
        }}

        Provide adjusted predictions for ALL {len(cluster_predictions)} clusters.
    """)

    try:
        llm_response = _call_llm(validation_prompt, llm_model)
        if llm_response is None:
            # Silently fall back to original predictions
            return predictions_df

        adjustments_data = json.loads(llm_response)
        adjustments = adjustments_data.get('adjustments', [])

        # Create adjustment mapping
        adjustment_map = {adj['cluster_id']: adj['adjusted_prediction'] for adj in adjustments}

        # Apply adjustments
        validated_df = predictions_df.copy()
        validated_df['p_adopt_original'] = validated_df['p_adopt']
        validated_df['p_adopt'] = validated_df['cluster_id'].map(
            lambda cid: np.clip(adjustment_map.get(cid, validated_df[validated_df['cluster_id'] == cid]['p_adopt'].iloc[0]), 0.0, 1.0)
        )

        return validated_df.sort_values('p_adopt', ascending=False)

    except (json.JSONDecodeError, KeyError, Exception) as e:
        # Silently fall back to original predictions
        return predictions_df

def predict_from_metadata(
    approved_metadata: dict,
    trained_models: dict,
    centroids: dict,
    embedder_model: object,
    cluster_summary_df: pd.DataFrame,
    use_llm_validation: bool = False,
    llm_model: str = None
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
        X_B = np.concatenate([centroid, content_embedding]).reshape(1, -1)
        scores['p_adopt'] = np.clip(path_b_model.predict(X_B)[0], 0.0, 1.0)
        cluster_scores.append(scores)

    scores_df = pd.DataFrame(cluster_scores)

    if not cluster_summary_df.empty:
        scores_df = scores_df.merge(cluster_summary_df[['cluster_id', 'users', 'cluster_name']], on='cluster_id', how='left')
        if 'users' in scores_df.columns:
            scores_df = scores_df.drop(columns=['users'])

    # Apply LLM validation if requested
    if use_llm_validation and llm_model:
        scores_df = llm_validate_predictions(scores_df, approved_metadata, llm_model)

    return scores_df.sort_values('p_adopt', ascending=False)

def compute_roi_value(pred_df: pd.DataFrame, user_counts: pd.DataFrame, avg_completion: float = AVERAGE_COMPLETION_RATE) -> float:
    """Compute predicted value from adoption predictions."""
    df = pred_df.merge(user_counts, on='cluster_id')
    expected_adopters = (df['p_adopt'] * df['users']).sum()
    value = expected_adopters * VALUE_PER_ADOPTER * avg_completion
    return value


def collect_talent_research(names: List[str], max_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Look up recent performance insights for notable talent using the Exa API."""
    clean_names = [name.strip() for name in names if name and name.strip()]
    if not clean_names:
        return {}

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        st.info("Set EXA_API_KEY to enable Notable Cast / Director research.")
        return {}

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key
    }
    search_url = "https://api.exa.ai/search"

    research: Dict[str, List[Dict[str, Any]]] = {}

    for name in clean_names:
        query = (
            f"{name} box office performance OR streaming audience impact OR viewership track record"
        )
        payload = {
            "query": query,
            "type": "auto",
            "numResults": max_results,
            "contents": {
                "text": True
            }
        }

        try:
            response = requests.post(search_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            raw_results = response.json().get("results", [])
        except Exception as exc:
            st.warning(f"Exa search failed for {name}: {exc}")
            research[name] = []
            continue

        normalized_results: List[Dict[str, Any]] = []
        for item in raw_results:
            title = item.get("title") or item.get("url") or "Untitled source"
            snippet = item.get("text") or item.get("snippet") or ""
            snippet = snippet.replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:497].rstrip() + "..."

            normalized_results.append({
                "title": title,
                "url": item.get("url"),
                "snippet": snippet,
                "score": item.get("score"),
                "publishedDate": item.get("publishedDate")
            })

        research[name] = normalized_results

    return research


def llm_estimate_talent_impact(
    metadata: Dict[str, Any],
    roi_baseline: Dict[str, Any],
    talent_research: Dict[str, List[Dict[str, Any]]],
    llm_model: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Estimate audience impact multipliers from notable talent using LLM synthesis."""
    if not talent_research:
        return None

    if not llm_model:
        st.warning("LLM model not configured; skipping talent impact synthesis.")
        return None

    # Build research digest for the prompt
    digest_sections = []
    for name, items in talent_research.items():
        if not items:
            digest_sections.append(f"{name}: no recent metrics surfaced.\n")
            continue

        bullet_points = []
        for item in items:
            title = item.get("title", "Untitled source")
            snippet = item.get("snippet", "")
            url = item.get("url") or ""
            if url:
                bullet_points.append(f"- {title}: {snippet} (Source: {url})")
            else:
                bullet_points.append(f"- {title}: {snippet}")

        digest_sections.append(f"{name}:\n" + "\n".join(bullet_points[:max(1, min(3, len(bullet_points)))]))

    research_digest = "\n\n".join(digest_sections)

    title = metadata.get('FRANCHISE_TITLE', 'Untitled Project')
    logline = metadata.get('logline', 'No logline provided.')
    baseline_value = roi_baseline.get('estimated_value', 0.0)
    baseline_roi = roi_baseline.get('roi_percentage', 0.0)
    budget = roi_baseline.get('estimated_cost', 0.0)

    prompt = dedent(f"""
        You are an entertainment analytics strategist. Given a project baseline ROI forecast and
        external research on lead actors or directors, estimate how notable talent might shift future performance.

        PROJECT CONTEXT:
        - Title: {title}
        - Content Type: {metadata.get('CONTENT_TYPE', 'Unknown')}
        - Logline: {logline}
        - Baseline gross value: ${baseline_value:,.0f}
        - Budget: ${budget:,.0f}
        - Baseline ROI: {baseline_roi:.1f}%

        TALENT RESEARCH (summaries of recent coverage, performance data, or comparable releases):
        {research_digest}

        TASK:
        1. Infer how the combined star power is likely to influence adoption or value.
        2. Express the effect as multipliers relative to the baseline estimate (1.0 = no change).
        3. Provide a low case, expected case, and high case multiplier and call out uncertainties.

        REQUIREMENTS:
        - Return ONLY valid JSON with this schema:
        {{
            "overall_multiplier_low": 0.95,
            "overall_multiplier_expected": 1.12,
            "overall_multiplier_high": 1.28,
            "confidence": "low|medium|high",
            "assumptions": ["String"],
            "talent_breakdown": [
                {{
                    "name": "Talent Name",
                    "impact_direction": "positive|neutral|negative",
                    "expected_change_pct": 12.5,
                    "low_change_pct": 5.0,
                    "high_change_pct": 20.0,
                    "rationale": "One-sentence justification referencing the research"
                }}
            ]
        }}
        - Multipliers must be numeric, between 0.5 and 1.75.
        - Percent fields represent percentage change relative to baseline (e.g., 12.5 = +12.5%).
        - Include at least one assumption noting gaps in the research sample.
        - If evidence suggests minimal impact, set all multipliers to 1.0.
    """)

    llm_response = _call_llm(prompt, llm_model)
    if not llm_response:
        return None

    try:
        parsed = json.loads(llm_response)
        return parsed
    except json.JSONDecodeError:
        st.warning("Could not parse talent impact response from LLM. Showing baseline only.")
        return None


def derive_talent_adjustments(roi_baseline: Dict[str, Any], impact_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate LLM output into concrete ROI adjustments."""
    if not impact_data:
        return None

    def _parse_multiplier(value: Any, default: float = 1.0) -> float:
        try:
            multiplier = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.5, min(1.75, multiplier))

    baseline_value = float(roi_baseline.get('estimated_value', 0.0))
    budget = float(roi_baseline.get('estimated_cost', 0.0))

    multiplier_low = _parse_multiplier(impact_data.get('overall_multiplier_low', 1.0))
    multiplier_expected = _parse_multiplier(impact_data.get('overall_multiplier_expected', 1.0))
    multiplier_high = _parse_multiplier(impact_data.get('overall_multiplier_high', 1.0))

    value_low = baseline_value * multiplier_low
    value_expected = baseline_value * multiplier_expected
    value_high = baseline_value * multiplier_high

    def _roi(value: float) -> float:
        if budget <= 0:
            return 0.0
        return ((value - budget) / budget) * 100

    adjustments = {
        'multiplier_low': multiplier_low,
        'multiplier_expected': multiplier_expected,
        'multiplier_high': multiplier_high,
        'value_low': value_low,
        'value_expected': value_expected,
        'value_high': value_high,
        'roi_low': _roi(value_low),
        'roi_expected': _roi(value_expected),
        'roi_high': _roi(value_high),
        'confidence': impact_data.get('confidence', 'medium'),
        'assumptions': impact_data.get('assumptions', []),
        'talent_breakdown': impact_data.get('talent_breakdown', [])
    }

    return adjustments


def parse_notable_talent(raw_input: Optional[str]) -> List[str]:
    """Convert user input into a deduplicated list of talent names."""
    if not raw_input:
        return []

    normalized = raw_input.replace('\n', ',').replace(';', ',')
    candidates = [name.strip() for name in normalized.split(',') if name and name.strip()]

    deduped: List[str] = []
    seen = set()
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)

    return deduped


def llm_estimate_talent_impact_engagement(
    metadata: Dict[str, Any],
    predictions_df: pd.DataFrame,
    talent_research: Dict[str, List[Dict[str, Any]]],
    llm_model: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Estimate engagement impact adjustments from notable talent using LLM synthesis."""
    if not talent_research:
        return None

    if not llm_model:
        st.warning("LLM model not configured; skipping talent impact synthesis.")
        return None

    # Build research digest for the prompt
    digest_sections = []
    for name, items in talent_research.items():
        if not items:
            digest_sections.append(f"{name}: no recent metrics surfaced.\n")
            continue

        bullet_points = []
        for item in items:
            title = item.get("title", "Untitled source")
            snippet = item.get("snippet", "")
            url = item.get("url") or ""
            if url:
                bullet_points.append(f"- {title}: {snippet} (Source: {url})")
            else:
                bullet_points.append(f"- {title}: {snippet}")

        digest_sections.append(f"{name}:\n" + "\n".join(bullet_points[:max(1, min(3, len(bullet_points)))]))

    research_digest = "\n\n".join(digest_sections)

    title = metadata.get('FRANCHISE_TITLE', 'Untitled Project')
    logline = metadata.get('logline', 'No logline provided.')

    # Prepare cluster predictions for LLM
    cluster_predictions = []
    for _, row in predictions_df.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_name = CLUSTER_LABELS.get(str(cluster_id), f"Cluster {cluster_id}")
        cluster_predictions.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'baseline_prediction': float(row['p_adopt'])
        })

    prompt = dedent(f"""
        You are an entertainment analytics strategist for BET+, a streaming platform focused on Black entertainment and culture.

        PROJECT CONTEXT:
        - Title: {title}
        - Content Type: {metadata.get('CONTENT_TYPE', 'Unknown')}
        - Logline: {logline}
        - Genre: {metadata.get('Genre', 'Unknown')}
        - Subgenre: {metadata.get('Subgenre', 'Unknown')}

        BASELINE ENGAGEMENT PREDICTIONS:
        Our ML model predicted these engagement probabilities for each audience cluster:
        {json.dumps(cluster_predictions, indent=2)}

        TALENT RESEARCH (summaries of recent coverage, performance data, or comparable releases):
        {research_digest}

        TASK:
        Based on the talent research, adjust the engagement predictions to reflect the star power and audience draw of the notable cast/directors.
        Consider:
        1. How the talent's previous work aligns with BET+ audience preferences
        2. Their track record with similar content
        3. Their cultural recognition and appeal to specific audience clusters
        4. Whether their presence would broaden appeal or deepen engagement with existing fans

        REQUIREMENTS:
        - Return ONLY valid JSON with this schema:
        {{
            "adjustments": [
                {{
                    "cluster_id": 0,
                    "adjusted_prediction": 0.75,
                    "multiplier": 1.15,
                    "rationale": "Brief explanation of adjustment"
                }},
                ...
            ],
            "overall_multiplier_low": 1.05,
            "overall_multiplier_expected": 1.15,
            "overall_multiplier_high": 1.25,
            "confidence": "low|medium|high",
            "assumptions": ["String"],
            "talent_breakdown": [
                {{
                    "name": "Talent Name",
                    "impact_direction": "positive|neutral|negative",
                    "expected_change_pct": 12.5,
                    "low_change_pct": 5.0,
                    "high_change_pct": 20.0,
                    "rationale": "One-sentence justification referencing the research"
                }}
            ]
        }}
        - Adjusted predictions must be between 0.0 and 1.0
        - Multipliers must be numeric, between 0.8 and 1.5
        - Percent fields represent percentage change relative to baseline
        - Provide adjustments for ALL {len(cluster_predictions)} clusters
        - If evidence suggests minimal impact, set all multipliers close to 1.0
    """)

    llm_response = _call_llm(prompt, llm_model)
    if not llm_response:
        return None

    try:
        parsed = json.loads(llm_response)
        return parsed
    except json.JSONDecodeError:
        st.warning("Could not parse talent impact response from LLM. Showing baseline only.")
        return None

def determine_budget_tier(budget: float, content_type: str) -> str:
    """Determine if this is a high-budget (top tier) investment based on industry standards."""
    # High-budget thresholds based on content type
    if content_type == "Feature Film":
        # Feature films: $50M+ is high-budget
        return "high" if budget >= 50_000_000 else "standard"
    else:
        # TV Shows: $5M+ per episode or $50M+ total is high-budget
        return "high" if budget >= 5_000_000 else "standard"

def predict_roi(metadata: dict, budget: float, content_type: str, trained_models: dict, centroids: dict, embedder_model: object, use_llm_validation: bool = False, llm_model: str = None) -> dict:
    """Predict ROI for a script given budget."""
    # Determine budget tier and set appropriate population total
    budget_tier = determine_budget_tier(budget, content_type)
    target_population = 3_000_000 if budget_tier == "high" else 600_000

    # Get predictions with appropriate population size
    user_counts = load_cluster_summary(target_total=target_population)
    if user_counts.empty:
        st.error("Could not load user data for ROI calculation")
        return None

    pred_df = predict_from_metadata(metadata, trained_models, centroids, embedder_model, user_counts, use_llm_validation, llm_model)

    # Calculate value using budget as the cost
    avg_completion_rate = AVERAGE_COMPLETION_RATE
    estimated_value = compute_roi_value(pred_df, user_counts, avg_completion=avg_completion_rate)
    estimated_cost = budget  # Use user's budget input as the cost

    # Calculate ROI
    net_profit = estimated_value - estimated_cost
    roi_percentage = (net_profit / estimated_cost) * 100 if estimated_cost > 0 else 0

    # Calculate expected adopters
    df_with_users = pred_df.merge(user_counts[['cluster_id', 'users']], on='cluster_id', how='left')
    df_with_users['users'] = df_with_users['users'].fillna(0)
    if 'cluster_name' in df_with_users.columns:
        df_with_users['cluster_name'] = df_with_users['cluster_name'].fillna(
            df_with_users['cluster_id'].astype(str).map(CLUSTER_LABELS)
        )
    else:
        df_with_users['cluster_name'] = df_with_users['cluster_id'].astype(str).map(CLUSTER_LABELS)

    df_with_users['expected_adopters'] = df_with_users['p_adopt'] * df_with_users['users']
    df_with_users['cluster_value'] = df_with_users['expected_adopters'] * VALUE_PER_ADOPTER * avg_completion_rate
    if estimated_value > 0:
        df_with_users['value_share'] = df_with_users['cluster_value'] / estimated_value
    else:
        df_with_users['value_share'] = 0.0

    expected_adopters = df_with_users['expected_adopters'].sum()
    cluster_breakdown = df_with_users.sort_values('cluster_value', ascending=False).reset_index(drop=True)

    return {
        'estimated_cost': estimated_cost,
        'estimated_value': estimated_value,
        'net_profit': net_profit,
        'roi_percentage': roi_percentage,
        'expected_adopters': expected_adopters,
        'pred_df': pred_df,
        'budget_tier': budget_tier,
        'target_population': target_population,
        'avg_completion_rate': avg_completion_rate,
        'cluster_breakdown': cluster_breakdown
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

# --- Password Authentication ---
def check_password():
    """Returns True if the user has entered the correct password."""

    # Initialize session state for authentication
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # If already authenticated, return True
    if st.session_state["password_correct"]:
        return True

    # Show password input
    st.markdown("""
    <div class="main-header">
        <h1>Signal</h1>
        <p>Audience engagement analysis for scripts and pitches</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Please enter password to access the application")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Login"):
        if password == "Flex123":
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False

# Check password before showing main app
if not check_password():
    st.stop()

# --- Streamlit App ---

# Modern header
st.markdown("""
<div class="main-header">
    <h1>Signal</h1>
    <p>Audience engagement analysis for scripts and pitches</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Engagement Predictor", "ROI Predictor", "Cluster Guide"])

# Initialize session state
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'roi_results' not in st.session_state:
    st.session_state.roi_results = None
if 'retry_feedback' not in st.session_state:
    st.session_state.retry_feedback = None
if 'script_text' not in st.session_state:
    st.session_state.script_text = None
if 'content_type' not in st.session_state:
    st.session_state.content_type = None
if 'roi_metadata' not in st.session_state:
    st.session_state.roi_metadata = None
if 'talent_results' not in st.session_state:
    st.session_state.talent_results = None
if 'engagement_talent_results' not in st.session_state:
    st.session_state.engagement_talent_results = None

llm_model = "x-ai/grok-4-fast"  # Hidden from UI - using OpenRouter

# Dialog for retry feedback
@st.dialog("Provide Feedback for Retry")
def retry_dialog():
    st.write("What would you like the AI to focus on or correct in the next attempt?")
    feedback = st.text_area(
        "Feedback",
        placeholder="e.g., 'The genre is wrong, it should be a thriller not a comedy' or 'Focus more on the main character's motivation'",
        height=150
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit & Retry", type="primary", width='stretch'):
            if feedback.strip():
                st.session_state.retry_feedback = feedback
                # Regenerate metadata with the stored script
                if st.session_state.script_text:
                    with st.spinner("Regenerating metadata with your feedback..."):
                        st.session_state.metadata = llm_define_metadata_v2(
                            st.session_state.script_text,
                            st.session_state.content_type,
                            llm_model,
                            retry_feedback=feedback
                        )
                    st.session_state.predictions = None
                    st.session_state.retry_feedback = None
                st.rerun()
            else:
                st.warning("Please provide feedback before retrying.")
    with col2:
        if st.button("Cancel", width='stretch'):
            st.rerun()

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
    # Dynamically set allowed file types based on PDF support
    allowed_types = ["txt", "pdf"] if HAS_PDF_SUPPORT else ["txt"]
    file_type_label = ".txt or .pdf" if HAS_PDF_SUPPORT else ".txt"

    uploaded_files = st.file_uploader(f"Upload script / pitch file(s) ({file_type_label})", type=allowed_types, key="engagement_upload", accept_multiple_files=True)

    if not HAS_PDF_SUPPORT and uploaded_files:
        st.info("PDF support is not available. Please upload .txt files only.")

    # Show alert for multiple files
    if uploaded_files and len(uploaded_files) > 1:
        st.info("Note: All uploaded documents should be for the same movie or series. They will be combined into a single analysis.")

    content_type = st.radio("Select Content Type", ["TV Show", "Feature Film"], horizontal=True)

    # LLM validation always enabled (hidden from user)
    use_llm_validation = True

    st.text_input(
        "Notable Cast / Director",
        placeholder="Issa Rae, Ava DuVernay",
        help="Optional. Add lead talent or directors to pull performance research and adjust engagement forecasts.",
        key="engagement_talent_input"
    )

    if uploaded_files and st.button("Generate Metadata"):
        script_text = ""

        # Process all uploaded files
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.pdf'):
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    script_text += pdf_text + "\n\n"
            else:
                # Read text file
                script_text += uploaded_file.read().decode("utf-8") + "\n\n"

        if script_text.strip():
            # Store script text and content type for potential retry
            st.session_state.script_text = script_text
            st.session_state.content_type = content_type

            with st.spinner("Generating metadata..."):
                st.session_state.metadata = llm_define_metadata_v2(
                    script_text,
                    content_type,
                    llm_model,
                    retry_feedback=st.session_state.retry_feedback
                )
            st.session_state.predictions = None
            # Clear retry feedback after use
            st.session_state.retry_feedback = None
        else:
            st.error("No text could be extracted from the uploaded files.")

    # --- 2. Review and Edit Details ---
    if st.session_state.metadata:
        st.subheader("Review and Edit Details")

        # Define user-friendly labels for each field
        field_labels = {
            'FRANCHISE_TITLE': 'Franchise Title',
            'logline': 'Logline',
            'Genre': 'Genre',
            'Subgenre': 'Subgenre',
            'CONTENT_TYPE': 'TV Show or Feature Film?',
            'Tonal_Comps': 'Similar Franchises / Tonal Comparables',
            'Shared_Tropes': 'Key Tropes or Themes',
            'Differential': 'Key Differentiators',
            'Protagonist_Demo': "Protagonist's Background / Backstory",
            'embedding_text': 'Summary'
        }

        edited_metadata = {}
        for key, value in st.session_state.metadata.items():
            label = field_labels.get(key, key)

            # Handle CONTENT_TYPE as dropdown
            if key == 'CONTENT_TYPE':
                options = ['TV Show', 'Feature Film']
                default_index = 0
                if isinstance(value, str):
                    if 'Feature' in value or 'film' in value.lower():
                        default_index = 1
                edited_metadata[key] = st.selectbox(label, options=options, index=default_index, key=f"engagement_{key}")
            # Make embedding_text read-only (disabled)
            elif key == 'embedding_text':
                st.text_area(label, value, key=f"engagement_{key}", disabled=True, help="Auto-generated summary of all fields above")
                edited_metadata[key] = value  # Keep original value
            # Handle other fields
            elif isinstance(value, list):
                edited_metadata[key] = st.text_area(label, ", ".join(map(str, value)), key=f"engagement_{key}").split(', ')
            else:
                edited_metadata[key] = st.text_area(label, value, key=f"engagement_{key}")

        st.session_state.metadata = edited_metadata

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Run Prediction"):
                with st.spinner("Running prediction..."):
                    # Get baseline predictions
                    baseline_predictions = predict_from_metadata(
                        st.session_state.metadata,
                        trained_models,
                        centroids,
                        embedder,
                        cluster_summary_df,
                        use_llm_validation=use_llm_validation,
                        llm_model=llm_model
                    )
                    st.session_state.predictions = baseline_predictions

                    # Check for notable talent
                    talent_names = parse_notable_talent(st.session_state.get('engagement_talent_input', ''))
                    if talent_names:
                        with st.spinner("Researching notable talent impact..."):
                            talent_research = collect_talent_research(talent_names)
                            talent_impact = None

                            if talent_research:
                                talent_impact = llm_estimate_talent_impact_engagement(
                                    st.session_state.metadata,
                                    baseline_predictions,
                                    talent_research,
                                    llm_model
                                )

                            # Store talent results
                            st.session_state.engagement_talent_results = {
                                'names': talent_names,
                                'research': talent_research,
                                'impact': talent_impact,
                                'baseline_predictions': baseline_predictions
                            }

                            # Apply adjustments to predictions if available
                            if talent_impact and 'adjustments' in talent_impact:
                                adjusted_predictions = baseline_predictions.copy()
                                adjustment_map = {adj['cluster_id']: adj['adjusted_prediction']
                                               for adj in talent_impact['adjustments']}
                                adjusted_predictions['p_adopt_baseline'] = adjusted_predictions['p_adopt']
                                adjusted_predictions['p_adopt'] = adjusted_predictions['cluster_id'].map(
                                    lambda cid: np.clip(adjustment_map.get(cid, adjusted_predictions[adjusted_predictions['cluster_id'] == cid]['p_adopt'].iloc[0]), 0.0, 1.0)
                                )
                                st.session_state.predictions = adjusted_predictions.sort_values('p_adopt', ascending=False)
                    else:
                        # Clear talent results if no talent specified
                        st.session_state.engagement_talent_results = None
        with col2:
            if st.button("🔄 Retry", help="Auto-generated details missed the mark? Try again."):
                retry_dialog()

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
        talent_results = st.session_state.get('engagement_talent_results')
        has_talent_impact = bool(talent_results and talent_results.get('impact') and talent_results['impact'].get('adjustments'))

        if has_talent_impact:
            baseline_predictions = talent_results.get('baseline_predictions')
            if baseline_predictions is not None:
                baseline_top_3 = baseline_predictions.sort_values('p_adopt', ascending=False).head(3)['p_adopt'].values
                baseline_engagement_index = baseline_top_3.mean()
                engagement_delta = engagement_index - baseline_engagement_index
                delta_label = f"{'+' if engagement_delta >= 0 else ''}{engagement_delta:.1%} vs baseline"
                st.metric("Engagement Score (w/ Talent)", f"{engagement_index:.0%}", delta=delta_label, help="Average engagement probability of top 3 audience clusters, adjusted for notable talent")
            else:
                st.metric("Engagement Score", f"{engagement_index:.0%}", help="Average engagement probability of top 3 audience clusters")
        else:
            st.metric("Engagement Score", f"{engagement_index:.0%}", help="Average engagement probability of top 3 audience clusters")

        # Display talent impact details if available
        if talent_results:
            talent_names = talent_results.get('names', [])
            talent_research = talent_results.get('research', {})
            talent_impact = talent_results.get('impact')

            if talent_names:
                st.markdown("---")

                if has_talent_impact:
                    st.subheader("Notable Cast / Director Impact")

                    talent_impact_data = talent_impact
                    multiplier_low = talent_impact_data.get('overall_multiplier_low', 1.0)
                    multiplier_expected = talent_impact_data.get('overall_multiplier_expected', 1.0)
                    multiplier_high = talent_impact_data.get('overall_multiplier_high', 1.0)

                    impact_cols = st.columns(3)
                    with impact_cols[0]:
                        st.metric("Low Case Multiplier", f"{multiplier_low:.2f}x")
                    with impact_cols[1]:
                        st.metric("Expected Multiplier", f"{multiplier_expected:.2f}x")
                    with impact_cols[2]:
                        st.metric("High Case Multiplier", f"{multiplier_high:.2f}x")

                    lift_expected = (multiplier_expected - 1.0) * 100
                    lift_low = (multiplier_low - 1.0) * 100
                    lift_high = (multiplier_high - 1.0) * 100
                    st.caption(
                        f"Talent-driven lift vs baseline: expected {lift_expected:+.1f}% (range {lift_low:+.1f}% to {lift_high:+.1f}%)."
                    )

                    breakdown = talent_impact_data.get('talent_breakdown') or []
                    if breakdown:
                        breakdown_df = pd.DataFrame(breakdown)
                        rename_map = {
                            'name': 'Talent',
                            'impact_direction': 'Direction',
                            'expected_change_pct': 'Expected %Δ',
                            'low_change_pct': 'Low %Δ',
                            'high_change_pct': 'High %Δ',
                            'rationale': 'Rationale'
                        }
                        breakdown_df = breakdown_df.rename(columns=rename_map)
                        pct_columns = ['Expected %Δ', 'Low %Δ', 'High %Δ']
                        def _fmt_pct(value: Any) -> str:
                            try:
                                return f"{float(value):+.1f}%"
                            except (TypeError, ValueError):
                                return "n/a"
                        for col in pct_columns:
                            if col in breakdown_df.columns:
                                breakdown_df[col] = breakdown_df[col].map(_fmt_pct)

                        st.dataframe(breakdown_df, width='stretch', hide_index=True)

                    assumptions = talent_impact_data.get('assumptions') or []
                    if assumptions:
                        st.markdown("**Assumptions & Notes**")
                        for item in assumptions:
                            st.markdown(f"- {item}")
                else:
                    if not talent_research:
                        st.info("Talent adjustments skipped. Add an EXA_API_KEY or refine the listed names to fetch research.")
                    else:
                        st.info("Talent research returned but the AI could not quantify impact. Baseline engagement shown above.")

                if talent_research:
                    with st.expander("Research sources (Exa)"):
                        for name in talent_names:
                            sources = talent_research.get(name, [])
                            st.markdown(f"**{name}**")
                            if not sources:
                                st.markdown("- No summaries returned.")
                                continue
                            for source in sources:
                                title = source.get('title', 'Source')
                                url = source.get('url')
                                snippet = source.get('snippet', '')
                                if url:
                                    st.markdown(f"- [{title}]({url}) – {snippet}")
                                else:
                                    st.markdown(f"- {title} – {snippet}")
                            st.markdown(" ")

        st.markdown("---")
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        # Detailed breakdown
        st.subheader("Audience Engagement Breakdown")
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        def truncate_label(text: str, max_length: int = 35) -> str:
            """Truncate text to max_length characters, adding ellipsis if needed."""
            if len(text) <= max_length:
                return text
            return text[:max_length-3] + "..."

        # Display top 3 clusters in their own row
        top_3_data = sorted_predictions.head(3)
        top_cols = st.columns(3)

        for idx, (i, row) in enumerate(top_3_data.iterrows()):
            cluster_id = str(row['cluster_id'])
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            truncated_label = truncate_label(label)
            p_adopt = row['p_adopt']

            # Get cluster description for tooltip
            cluster_info = CLUSTER_INFO.get(int(cluster_id), {})
            description = cluster_info.get('description', label if len(label) > 35 else None)

            with top_cols[idx]:
                st.markdown(f"<div style='margin-bottom: 10px;'><strong>{idx + 1}. {truncated_label}</strong></div>", unsafe_allow_html=True)
                st.metric(label="", value=f"{p_adopt:.0%}", help=description)

        # Add blank row for spacing
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        # Create a 3x4 grid for all clusters
        cols = st.columns(4)

        for i, row in enumerate(sorted_predictions.itertuples()):
            cluster_id = str(row.cluster_id)
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            truncated_label = truncate_label(label)
            p_adopt = row.p_adopt

            # Get cluster description for tooltip
            cluster_info = CLUSTER_INFO.get(int(cluster_id), {})
            description = cluster_info.get('description', label if len(label) > 35 else None)

            with cols[i % 4]:
                st.metric(label=truncated_label, value=f"{p_adopt:.0%}", help=description)

        # Optional: Show detailed table in expander
        with st.expander("View Detailed Data"):
            # Add cluster names and hide p_adopt_original
            display_df = st.session_state.predictions.copy()
            display_df['Taste Cluster Name'] = display_df['cluster_id'].astype(str).map(CLUSTER_LABELS)
            display_df['Probability of High Engagement'] = display_df['p_adopt']

            # Select and reorder columns (drop cluster_id)
            display_df = display_df[['Taste Cluster Name', 'Probability of High Engagement']]

            st.dataframe(display_df, width='stretch', hide_index=True)

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
    # Dynamically set allowed file types based on PDF support
    roi_allowed_types = ["txt", "pdf"] if HAS_PDF_SUPPORT else ["txt"]
    roi_file_type_label = ".txt or .pdf" if HAS_PDF_SUPPORT else ".txt"

    roi_uploaded_files = st.file_uploader(f"Upload script / pitch file(s) ({roi_file_type_label})", type=roi_allowed_types, key="roi_upload", accept_multiple_files=True)

    if not HAS_PDF_SUPPORT and roi_uploaded_files:
        st.info("PDF support is not available. Please upload .txt files only.")

    # Show alert for multiple files
    if roi_uploaded_files and len(roi_uploaded_files) > 1:
        st.info("Note: All uploaded documents should be for the same movie or series. They will be combined into a single analysis.")

    roi_content_type = st.radio("Select Content Type", ["TV Show", "Feature Film"], horizontal=True, key="roi_content_type")

    # LLM validation always enabled (hidden from user)
    roi_use_llm_validation = True

    # --- 2. Budget Input ---
    st.markdown("#### Budget")
    budget_col1, budget_col2 = st.columns(2)
    with budget_col1:
        acquisition_budget = st.number_input(
            "Acquisition Budget ($)",
            min_value=0,
            value=100000,
            step=10000,
            help="Cost to acquire or produce the content"
        )
    with budget_col2:
        marketing_budget = st.number_input(
            "Marketing Budget ($)",
            min_value=0,
            value=0,
            step=10000,
            help="Budget allocated for marketing and promotion"
        )

    # Calculate total budget
    total_budget = acquisition_budget + marketing_budget
    if total_budget > 0:
        st.caption(f"Total Budget: ${total_budget:,}")

    st.text_input(
        "Notable Cast / Director",
        placeholder="Issa Rae, Ava DuVernay",
        help="Optional. Add lead talent or directors to pull performance research and adjust ROI forecasts.",
        key="roi_talent_input"
    )

    analyze_clicked = st.button("Analyze ROI", key="roi_analyze")

    if roi_uploaded_files and analyze_clicked:
        script_text = ""

        # Process all uploaded files
        for uploaded_file in roi_uploaded_files:
            if uploaded_file.name.endswith('.pdf'):
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    script_text += pdf_text + "\n\n"
            else:
                # Read text file
                script_text += uploaded_file.read().decode("utf-8") + "\n\n"

        if script_text.strip():
            with st.spinner("Analyzing script and calculating ROI..."):
                # Generate metadata
                roi_metadata = llm_define_metadata_v2(script_text, roi_content_type, llm_model)
                if roi_metadata:
                    st.session_state.roi_metadata = roi_metadata

                    results = predict_roi(
                        roi_metadata,
                        total_budget,
                        roi_content_type,
                        trained_models,
                        centroids,
                        embedder,
                        use_llm_validation=roi_use_llm_validation,
                        llm_model=llm_model
                    )

                    # Add budget breakdown to results
                    if results:
                        results['acquisition_budget'] = acquisition_budget
                        results['marketing_budget'] = marketing_budget

                    if results:
                        talent_names = parse_notable_talent(st.session_state.get('roi_talent_input', ''))

                        talent_research = collect_talent_research(talent_names) if talent_names else {}
                        talent_impact_raw = None
                        talent_adjustments = None

                        if talent_research:
                            talent_impact_raw = llm_estimate_talent_impact(
                                roi_metadata,
                                results,
                                talent_research,
                                llm_model
                            )
                            if talent_impact_raw:
                                talent_adjustments = derive_talent_adjustments(results, talent_impact_raw)

                        if talent_adjustments:
                            # Recalculate cluster breakdown with talent-adjusted values
                            adjusted_breakdown = results['cluster_breakdown'].copy()
                            multiplier_expected = talent_adjustments['multiplier_expected']

                            # Apply multiplier to adopters and recalculate value
                            adjusted_breakdown['expected_adopters'] = adjusted_breakdown['expected_adopters'] * multiplier_expected
                            adjusted_breakdown['cluster_value'] = adjusted_breakdown['expected_adopters'] * VALUE_PER_ADOPTER * results['avg_completion_rate']

                            # Recalculate value share based on new total
                            total_adjusted_value = adjusted_breakdown['cluster_value'].sum()
                            if total_adjusted_value > 0:
                                adjusted_breakdown['value_share'] = adjusted_breakdown['cluster_value'] / total_adjusted_value

                            # Re-sort by adjusted cluster value
                            adjusted_breakdown = adjusted_breakdown.sort_values('cluster_value', ascending=False).reset_index(drop=True)

                            results.update({
                                'estimated_value_adjusted': talent_adjustments['value_expected'],
                                'estimated_value_low': talent_adjustments['value_low'],
                                'estimated_value_high': talent_adjustments['value_high'],
                                'roi_percentage_adjusted': talent_adjustments['roi_expected'],
                                'roi_percentage_low': talent_adjustments['roi_low'],
                                'roi_percentage_high': talent_adjustments['roi_high'],
                                'net_profit_adjusted': talent_adjustments['value_expected'] - results['estimated_cost'],
                                'expected_adopters_adjusted': results['expected_adopters'] * talent_adjustments['multiplier_expected'],
                                'expected_adopters_low': results['expected_adopters'] * talent_adjustments['multiplier_low'],
                                'expected_adopters_high': results['expected_adopters'] * talent_adjustments['multiplier_high'],
                                'talent_multipliers': {
                                    'low': talent_adjustments['multiplier_low'],
                                    'expected': talent_adjustments['multiplier_expected'],
                                    'high': talent_adjustments['multiplier_high']
                                },
                                'talent_assumptions': talent_adjustments.get('assumptions', []),
                                'talent_breakdown': talent_adjustments.get('talent_breakdown', []),
                                'cluster_breakdown_adjusted': adjusted_breakdown,
                                'cluster_breakdown_baseline': results['cluster_breakdown']
                            })
                        else:
                            results.update({
                                'estimated_value_adjusted': None,
                                'estimated_value_low': None,
                                'estimated_value_high': None,
                                'roi_percentage_adjusted': None,
                                'roi_percentage_low': None,
                                'roi_percentage_high': None,
                                'net_profit_adjusted': None,
                                'expected_adopters_adjusted': None,
                                'expected_adopters_low': None,
                                'expected_adopters_high': None,
                                'talent_multipliers': None,
                                'talent_assumptions': None,
                                'talent_breakdown': None
                            })

                        results['talent_names'] = talent_names
                        results['talent_research'] = talent_research
                        results['talent_impact_raw'] = talent_impact_raw
                        results['talent_adjustments'] = talent_adjustments

                        st.session_state.talent_results = {
                            'names': talent_names,
                            'research': talent_research,
                            'impact': talent_impact_raw,
                            'adjustments': talent_adjustments
                        }

                        st.session_state.roi_results = results
                    else:
                        st.error("Failed to calculate ROI.")
                else:
                    st.error("Failed to generate metadata from script")
        else:
            st.error("No text could be extracted from the uploaded files.")

    # --- 3. Display ROI Results ---
    if st.session_state.roi_results is not None:
        results = st.session_state.roi_results

        talent_adjustments = results.get('talent_adjustments')
        has_talent_impact = bool(talent_adjustments and results.get('estimated_value_adjusted'))
        talent_names = results.get('talent_names', [])
        talent_research = results.get('talent_research', {})

        base_value = float(results.get('estimated_value', 0.0))
        base_roi = float(results.get('roi_percentage', 0.0))
        base_net_profit = float(results.get('net_profit', 0.0))
        base_adopters = float(results.get('expected_adopters', 0.0))

        def _resolve(value, fallback):
            return float(value) if value is not None else float(fallback)

        value_display = _resolve(results.get('estimated_value_adjusted') if has_talent_impact else None, base_value)
        roi_display = _resolve(results.get('roi_percentage_adjusted') if has_talent_impact else None, base_roi)
        net_profit_display = _resolve(results.get('net_profit_adjusted') if has_talent_impact else None, base_net_profit)
        adopters_display = _resolve(results.get('expected_adopters_adjusted') if has_talent_impact else None, base_adopters)

        roi_for_recommendation = roi_display

        if roi_for_recommendation > 50:
            roi_recommendation = "Strong Investment"
            roi_color = "green"
        elif roi_for_recommendation > 0:
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
        value_delta_label = None
        roi_delta_label = None
        if has_talent_impact:
            diff_value = value_display - base_value
            diff_roi = roi_display - base_roi
            if abs(diff_value) >= 1:
                value_delta_label = f"{'+' if diff_value >= 0 else '-'}${abs(diff_value):,.0f} vs baseline"
            if abs(diff_roi) >= 0.1:
                roi_delta_label = f"{'+' if diff_roi >= 0 else '-'}{abs(diff_roi):.1f} pts vs baseline"

        col1, col2, col3 = st.columns(3)
        with col1:
            acquisition_budget = results.get('acquisition_budget', 0)
            marketing_budget = results.get('marketing_budget', 0)
            total_cost = results['estimated_cost']

            # Build help text with breakdown
            if marketing_budget > 0:
                cost_help = f"Acquisition: ${acquisition_budget:,} | Marketing: ${marketing_budget:,}"
            else:
                cost_help = f"Acquisition: ${acquisition_budget:,}"

            st.metric("Estimated Cost", f"${total_cost:,.0f}", help=cost_help)
        with col2:
            st.metric("Estimated Value", f"${value_display:,.0f}", delta=value_delta_label)
        with col3:
            st.metric("ROI", f"{roi_display:.1f}%", delta=roi_delta_label)

        st.markdown("---")

        # Additional Details
        profit_delta_label = None
        adopters_delta_label = None
        if has_talent_impact:
            diff_profit = net_profit_display - base_net_profit
            diff_adopters = adopters_display - base_adopters
            if abs(diff_profit) >= 1:
                profit_delta_label = f"{'+' if diff_profit >= 0 else '-'}${abs(diff_profit):,.0f} vs baseline"
            if abs(diff_adopters) >= 1:
                adopters_delta_label = f"{'+' if diff_adopters >= 0 else '-'}{abs(diff_adopters):,.0f} vs baseline"

        col4, col5 = st.columns(2)
        with col4:
            st.metric("Net Profit", f"${net_profit_display:,.0f}", delta=profit_delta_label)
        with col5:
            st.metric("Expected Adopters", f"{adopters_display:,.0f}", delta=adopters_delta_label)

        if has_talent_impact:
            st.caption(
                f"Baseline (pre-talent) forecast: value ${base_value:,.0f}, ROI {base_roi:.1f}%, net profit ${base_net_profit:,.0f}."
            )

        # Show budget and population tier information (removed for cleaner display)

        if talent_names:
            st.markdown("---")

            if has_talent_impact:
                st.subheader("Notable Cast / Director Impact")

                impact_cols = st.columns(3)
                with impact_cols[0]:
                    st.metric("Low Case Value", f"${talent_adjustments['value_low']:,.0f}")
                with impact_cols[1]:
                    st.metric("Expected Value", f"${talent_adjustments['value_expected']:,.0f}")
                with impact_cols[2]:
                    st.metric("High Case Value", f"${talent_adjustments['value_high']:,.0f}")

                lift_expected = (talent_adjustments['multiplier_expected'] - 1.0) * 100
                lift_low = (talent_adjustments['multiplier_low'] - 1.0) * 100
                lift_high = (talent_adjustments['multiplier_high'] - 1.0) * 100
                st.caption(
                    f"Talent-driven lift vs baseline: expected {lift_expected:+.1f}% (range {lift_low:+.1f}% to {lift_high:+.1f}%)."
                )

                breakdown = talent_adjustments.get('talent_breakdown') or []
                if breakdown:
                    breakdown_df = pd.DataFrame(breakdown)
                    rename_map = {
                        'name': 'Talent',
                        'impact_direction': 'Direction',
                        'expected_change_pct': 'Expected %Δ',
                        'low_change_pct': 'Low %Δ',
                        'high_change_pct': 'High %Δ',
                        'rationale': 'Rationale'
                    }
                    breakdown_df = breakdown_df.rename(columns=rename_map)
                    pct_columns = ['Expected %Δ', 'Low %Δ', 'High %Δ']
                    def _fmt_pct(value: Any) -> str:
                        try:
                            return f"{float(value):+.1f}%"
                        except (TypeError, ValueError):
                            return "n/a"
                    for col in pct_columns:
                        if col in breakdown_df.columns:
                            breakdown_df[col] = breakdown_df[col].map(_fmt_pct)

                    st.dataframe(breakdown_df, width='stretch', hide_index=True)

                assumptions = talent_adjustments.get('assumptions') or []
                if assumptions:
                    st.markdown("**Assumptions & Notes**")
                    for item in assumptions:
                        st.markdown(f"- {item}")
            else:
                if not talent_research:
                    st.info("Talent adjustments skipped. Add an EXA_API_KEY or refine the listed names to fetch research.")
                else:
                    st.info("Talent research returned but the AI could not quantify impact. Baseline ROI shown above.")

            if talent_research:
                with st.expander("Research sources (Exa)"):
                    for name in talent_names:
                        sources = talent_research.get(name, [])
                        st.markdown(f"**{name}**")
                        if not sources:
                            st.markdown("- No summaries returned.")
                            continue
                        for source in sources:
                            title = source.get('title', 'Source')
                            url = source.get('url')
                            snippet = source.get('snippet', '')
                            if url:
                                st.markdown(f"- [{title}]({url}) – {snippet}")
                            else:
                                st.markdown(f"- {title} – {snippet}")
                        st.markdown(" ")

        st.markdown("---")

        # Cluster Breakdown
        st.subheader("Value by Audience Cluster")

        # Use adjusted breakdown if talent impact exists, otherwise use baseline
        has_adjusted_breakdown = bool(talent_adjustments and results.get('cluster_breakdown_adjusted') is not None)
        cluster_breakdown = results.get('cluster_breakdown_adjusted') if has_adjusted_breakdown else results.get('cluster_breakdown')

        if cluster_breakdown is not None and not cluster_breakdown.empty:
            total_value = float(cluster_breakdown['cluster_value'].sum())
            total_share = float(cluster_breakdown['value_share'].sum())

            # Show comparison if we have adjusted values
            if has_adjusted_breakdown:
                baseline_breakdown = results.get('cluster_breakdown_baseline')
                baseline_total = float(baseline_breakdown['cluster_value'].sum()) if baseline_breakdown is not None else 0
                value_delta = total_value - baseline_total
                delta_label = f"{'+' if value_delta >= 0 else ''}{value_delta:,.0f} vs baseline" if baseline_total > 0 else None
            else:
                delta_label = None

            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.metric("Cluster Value Sum", f"${total_value:,.0f}", delta=delta_label)
            with summary_cols[1]:
                st.metric("Value Coverage", f"{total_share:.0%}")

            if has_adjusted_breakdown:
                st.caption("Breakdown reflects talent-adjusted values. Baseline breakdown available in detailed view below.")

            # Removed validation messaging for cleaner display

            # Display top value clusters
            cols = st.columns(4)
            for i, row in enumerate(cluster_breakdown.head(12).itertuples()):
                with cols[i % 4]:
                    st.metric(
                        label=row.cluster_name if hasattr(row, 'cluster_name') else f"Cluster {row.cluster_id}",
                        value=f"${row.cluster_value:,.0f}",
                        help=(
                            f"P(adopt): {row.p_adopt:.1%}, Share: {row.value_share:.1%}"
                        )
                    )

        # Detailed table
        with st.expander("View Detailed Breakdown"):
            if cluster_breakdown is not None and not cluster_breakdown.empty:
                if has_adjusted_breakdown:
                    # Show both baseline and adjusted in tabs
                    tab_adjusted, tab_baseline = st.tabs(["Adjusted (w/ Talent)", "Baseline"])

                    with tab_adjusted:
                        display_columns = ['cluster_name', 'p_adopt', 'cluster_value', 'value_share']
                        display_breakdown = cluster_breakdown[display_columns].copy()
                        display_breakdown = display_breakdown.rename(columns={
                            'cluster_name': 'Cluster',
                            'p_adopt': 'Engagement %',
                            'cluster_value': 'Value',
                            'value_share': 'Value Share'
                        })
                        display_breakdown['Engagement %'] = display_breakdown['Engagement %'].map(lambda x: f"{x:.1%}")
                        display_breakdown['Value'] = display_breakdown['Value'].map(lambda x: f"${x:,.0f}")
                        display_breakdown['Value Share'] = display_breakdown['Value Share'].map(lambda x: f"{x:.1%}")
                        st.dataframe(display_breakdown, width='stretch', hide_index=True)

                    with tab_baseline:
                        baseline_breakdown = results.get('cluster_breakdown_baseline')
                        if baseline_breakdown is not None and not baseline_breakdown.empty:
                            display_columns = ['cluster_name', 'p_adopt', 'cluster_value', 'value_share']
                            display_baseline = baseline_breakdown[display_columns].copy()
                            display_baseline = display_baseline.rename(columns={
                                'cluster_name': 'Cluster',
                                'p_adopt': 'Engagement %',
                                'cluster_value': 'Value',
                                'value_share': 'Value Share'
                            })
                            display_baseline['Engagement %'] = display_baseline['Engagement %'].map(lambda x: f"{x:.1%}")
                            display_baseline['Value'] = display_baseline['Value'].map(lambda x: f"${x:,.0f}")
                            display_baseline['Value Share'] = display_baseline['Value Share'].map(lambda x: f"{x:.1%}")
                            st.dataframe(display_baseline, width='stretch', hide_index=True)
                else:
                    display_columns = ['cluster_name', 'p_adopt', 'cluster_value', 'value_share']
                    display_breakdown = cluster_breakdown[display_columns].copy()
                    display_breakdown = display_breakdown.rename(columns={
                        'cluster_name': 'Cluster',
                        'p_adopt': 'Engagement %',
                        'cluster_value': 'Value',
                        'value_share': 'Value Share'
                    })
                    display_breakdown['Engagement %'] = display_breakdown['Engagement %'].map(lambda x: f"{x:.1%}")
                    display_breakdown['Value'] = display_breakdown['Value'].map(lambda x: f"${x:,.0f}")
                    display_breakdown['Value Share'] = display_breakdown['Value Share'].map(lambda x: f"{x:.1%}")
                    st.dataframe(display_breakdown, use_container_width=True, hide_index=True)

# ==================== TAB 3: CLUSTER GUIDE ====================
with tab3:
    st.markdown("### Understanding Our Audience Clusters")
    st.markdown("Our audience is organized into 12 distinct taste clusters, each with unique content preferences and viewing patterns.")

    # Create expandable sections for each cluster
    for cluster_id in range(12):
        cluster = CLUSTER_INFO[cluster_id]

        with st.expander(f"**Cluster {cluster_id}: {cluster['name']}**"):
            # Subtitle
            st.markdown(f"### {cluster['subtitle']}")

            # Description
            st.markdown(f"{cluster['description']}")

            st.markdown("---")

            # Create three columns for organized display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Dominant Genres**")
                for genre in cluster['genres']:
                    st.markdown(f"- {genre}")

            with col2:
                st.markdown("**Dominant Subgenres**")
                for subgenre in cluster['subgenres']:
                    st.markdown(f"- {subgenre}")

            with col3:
                st.markdown("**Common Tropes**")
                for trope in cluster['tropes']:
                    st.markdown(f"- {trope}")

    # Add summary section at the bottom
    st.markdown("---")
    st.markdown("### Quick Reference")
    st.markdown("Use this guide to understand which audience segments your content might resonate with most strongly.")

    # Create a summary table
    summary_data = []
    for cluster_id in range(12):
        cluster = CLUSTER_INFO[cluster_id]
        summary_data.append({
            "Cluster": f"{cluster_id}: {cluster['name']}",
            "Primary Focus": cluster['subtitle'],
            "Top Genres": ", ".join(cluster['genres'][:2])
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width='stretch', hide_index=True)
