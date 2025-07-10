import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- I. Data and Model Loading ---

@st.cache_data
def load_and_process_data(csv_path):
    """
    Loads data, engineers features, and computes the recommendation model components.
    The decorator caches the output for improved performance.
    """
    df = pd.read_csv(csv_path)
    df['soup'] = df['series_name'].fillna('') + ' ' + \
                 df['publisher'].fillna('') + ' ' + \
                 df['description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.7)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# --- II. Core Recommender Logic ---

def get_recommendations(title, data, cosine_sim_matrix):
    """
    Finds the top 10 most similar comics for a given title.
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
    return data.iloc[comic_indices]

# --- III. Streamlit Application UI ---

st.set_page_config(layout="wide")
st.title('📚 Indie Comic Book Recommender')

# Initialize session state to remember the last clicked comic.
if 'selected_comic_details' not in st.session_state:
    st.session_state.selected_comic_details = None

# Load data and build model components.
try:
    df, cosine_sim = load_and_process_data('comicvine_comics_final.csv')

    # Create two tabs to organize the UI.
    tab1, tab2 = st.tabs(["Recommender", "Data Insights"])

    # --- Recommender Tab ---
    with tab1:
        st.header('Discover Your Next Favorite Series')
        comic_titles = df['series_name'].sort_values().tolist()
        user_input = st.selectbox('Enter or select a comic book series you like:', options=comic_titles)

        if st.button('Get Recommendations'):
            # When the main button is clicked, get recommendations but clear any old details.
            st.session_state.selected_comic_details = None 
            recommendations = get_recommendations(user_input, df, cosine_sim)
            # Store the full recommendations in session state to persist them.
            st.session_state.recommendations = recommendations

    # --- Display Recommendations and Details in a Two-Column Layout ---
    # This part of the code is now outside the button 'if' block to ensure it always runs.
    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        
        # Create two columns for the master-detail view.
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Recommendations:")
            # Display each recommendation as a clickable button.
            for index, row in st.session_state.recommendations.iterrows():
                if st.button(row['series_name'], key=f"rec_{index}"):
                    # When a button is clicked, store its details in session state.
                    st.session_state.selected_comic_details = row
        
        with col2:
            # Check if a comic has been selected to display its details.
            if st.session_state.selected_comic_details is not None:
                details = st.session_state.selected_comic_details
                st.subheader(details['series_name'])
                st.write(f"**Publisher:** {details['publisher']}")
                st.write(f"**Start Year:** {int(details['start_year'])}")
                st.write(f"**Number of Issues:** {int(details['issue_count'])}")
                st.write("**Description:**")
                st.write(details['description'])
            else:
                # Show a placeholder if no comic is selected yet.
                st.info("Click on a recommendation to see more details here.")

    # --- Data Insights Tab ---
    with tab2:
        st.header('About the Dataset')
        st.subheader("Top 20 Independent Publishers")
        st.image('images/top-20-publishers.png', caption='This chart shows the number of unique series from the top 20 publishers in the dataset.')
        st.subheader("Most Common Genres")
        st.image('images/most-common-genres.png', caption='This chart shows the frequency of genre keywords found within the descriptions of the comics.')
        st.subheader("Indie Series Launched per Decade")
        st.image('images/indie-series-launched-per-decade.png', caption='This chart shows the trend of new indie series being launched over the decades.')

except FileNotFoundError:
    st.error("`comicvine_comics_final.csv` not found. Please make sure it's in the same directory as `app.py`.")