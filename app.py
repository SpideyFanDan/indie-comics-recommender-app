import streamlit as st
import pandas as pd
import pickle

# --- DATA AND MODEL LOADING ---

# Use st.cache_data to load the data and model only once, making the app faster.
@st.cache_data
def load_model_components():
    """Load the CSV, TF-IDF vectorizer, and cosine similarity matrix."""
    try:
        # --- THIS LINE IS CHANGED ---
        # Load the DataFrame directly from the CSV file.
        df = pd.read_csv('comicvine_comics.csv')
        
        # Keep loading the other two files with pickle.
        tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
        return df, tfidf_vectorizer, cosine_sim
    except FileNotFoundError:
        st.error("Model or data files not found. Please make sure the .pkl and .csv files are in the correct directory.")
        return None, None, None

# Load the components
df, tfidf_vectorizer, cosine_sim = load_model_components()

# --- RECOMMENDATION FUNCTION ---

# This is the same function we built and tested in the Colab notebook.
def get_recommendations(title, cosine_sim_matrix, data):
    """
    This function takes a comic title and returns a list of the 10 most similar comics.
    """
    # Create a Series of indices and titles for quick lookups
    indices = pd.Series(data.index, index=data['series_name']).drop_duplicates()
    
    # Get the index of the comic that matches the title
    try:
        idx = indices[title]
    except KeyError:
        return f"Comic titled '{title}' not found in the dataset."

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the comics based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar comics (excluding itself)
    sim_scores = sim_scores[1:11]

    # Get the comic indices
    comic_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 comics
    recommendations = data['series_name'].iloc[comic_indices]
    return recommendations

# --- STREAMLIT APP UI ---

st.title('📚 Indie Comic Book Recommender')
st.header('Discover Your Next Favorite Series')

# Create a text input box for the user
user_input = st.text_input('Enter a comic book series you like:', 'Saga')

# Create a button to get recommendations
if st.button('Get Recommendations'):
    if df is not None:
        # Get and display recommendations
        recommendations = get_recommendations(user_input, cosine_sim, df)
        
        if isinstance(recommendations, str):
            st.warning(recommendations) # Show a warning if the comic is not found
        else:
            st.subheader('Here are some series you might enjoy:')
            for title in recommendations:
                st.write(f"- {title}")
    else:
        st.error("Could not load model. Please check the file paths.")