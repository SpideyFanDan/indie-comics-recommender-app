import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- DATA LOADING AND MODEL CREATION ---

# This function loads the data, creates the TF-IDF matrix, and cosine similarity matrix.
# The @st.cache_data decorator ensures this expensive computation only runs once.
@st.cache_data
def load_and_process_data(csv_path):
    """
    Loads data, cleans it, and creates the recommendation model components.
    Returns the DataFrame and the cosine similarity matrix.
    """
    # Load the DataFrame from the CSV file.
    df = pd.read_csv(csv_path)
    
    # Preprocess the data
    df['description'] = df['description'].fillna('')
    
    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.7)
    
    # Create the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

# --- RECOMMENDATION FUNCTION ---

def get_recommendations(title, data, cosine_sim_matrix):
    """
    This function takes a comic title and returns a list of the 10 most similar comics.
    """
    indices = pd.Series(data.index, index=data['series_name']).drop_duplicates()
    
    try:
        idx = indices[title]
    except KeyError:
        return f"Comic titled '{title}' not found in the dataset."

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    comic_indices = [i[0] for i in sim_scores]
    
    return data['series_name'].iloc[comic_indices]

# --- STREAMLIT APP ---

st.title('📚 Indie Comic Book Recommender')
st.header('Discover Your Next Favorite Series')

# Load data and build model
try:
    df, cosine_sim = load_and_process_data('comicvine_comics.csv')

    # Create a list of comic titles for the select box
    comic_titles = df['series_name'].sort_values().tolist()

    # Create a select box for user input
    user_input = st.selectbox('Enter or select a comic book series you like:', options=comic_titles)

    if st.button('Get Recommendations'):
        recommendations = get_recommendations(user_input, df, cosine_sim)
        
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.subheader(f"Recommendations based on '{user_input}':")
            for title in recommendations:
                st.write(f"- {title}")

except FileNotFoundError:
    st.error("`comicvine_comics.csv` not found. Please make sure it's in the same directory as `app.py`.")