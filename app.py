import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- DATA LOADING AND MODEL CREATION ---

@st.cache_data
def load_and_process_data(csv_path):
    """
    Loads data, creates a combined text 'soup', and builds the recommendation model.
    """
    df = pd.read_csv(csv_path)
    
    # --- THIS IS THE KEY IMPROVEMENT ---
    # Create a 'soup' of text features to get better recommendations.
    # We fill any missing values with an empty string first.
    df['soup'] = df['series_name'].fillna('') + ' ' + \
                 df['publisher'].fillna('') + ' ' + \
                 df['description'].fillna('')

    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.7)
    
    # Create the TF-IDF matrix from the new 'soup' column
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
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

try:
    df, cosine_sim = load_and_process_data('comicvine_comics.csv')

    comic_titles = df['series_name'].sort_values().tolist()
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