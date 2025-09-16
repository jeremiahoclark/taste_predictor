# Overview

This is a machine learning-powered taste prediction application built with Streamlit. The system analyzes user preferences and uses clustering algorithms to predict and recommend content based on taste profiles. It integrates with Groq's LLM API for enhanced recommendation capabilities and uses sentence transformers for embedding-based predictions.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Single-page interactive web interface for user input and results display
- **Plotly Visualizations**: Interactive charts and graphs for data visualization and cluster analysis
- **Responsive UI**: Built-in Streamlit components for forms, file uploads, and data display

## Backend Architecture
- **Machine Learning Pipeline**: Scikit-learn based clustering and prediction models
- **Model Persistence**: Joblib for saving and loading trained ML models
- **Data Processing**: Pandas and NumPy for data manipulation and analysis
- **Embedding Generation**: Sentence Transformers for converting text to vector embeddings

## Prediction System
- **Cluster-Based Recommendations**: Pre-trained clustering model with 12 distinct taste clusters
- **Hardcoded Cluster Labels**: Predefined taste categories including romance_thriller, animated_comedy, reality shows, etc.
- **LLM Integration**: Groq API for enhanced natural language processing and recommendation explanations
- **Fallback Mechanisms**: Graceful handling of missing dependencies with warning messages

## Configuration Management
- **Environment Variables**: python-dotenv for secure API key management
- **Resource Caching**: Streamlit's @st.cache_resource decorator for model loading optimization
- **Dependency Management**: Optional imports with feature degradation for missing packages

# External Dependencies

## Machine Learning Services
- **Groq API**: Large language model integration for enhanced recommendations
- **Sentence Transformers**: HuggingFace model (all-MiniLM-L6-v2) for text embeddings
- **Scikit-learn**: Core machine learning algorithms and clustering

## Data Processing
- **Pandas**: Data manipulation and analysis framework
- **NumPy**: Numerical computing and array operations
- **Joblib**: Model serialization and persistence

## Web Framework
- **Streamlit**: Web application framework and deployment platform
- **Plotly**: Interactive data visualization library

## Configuration
- **python-dotenv**: Environment variable management for API keys and secrets