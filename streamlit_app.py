import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                         usecols=[0, 1], names=['item_id', 'title'])
    df = pd.merge(ratings, movies, on='item_id')

    # Content-based (title + genres)
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    full_movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                              names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols)
    full_movies['genres'] = full_movies[genre_cols].apply(
        lambda row: ' '.join([g for g, val in row.items() if val == 1]), axis=1)
    movies_meta = full_movies[['item_id', 'title', 'genres']]
    movies_meta['text'] = movies_meta['title'] + ' ' + movies_meta['genres']

    return df, ratings, movies, movies_meta

df, ratings, movies, movies_meta = load_data()

# Precompute matrices for filtering
user_movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
user_ratings = user_movie_matrix.sub(user_movie_matrix.mean(axis=1), axis=0)
user_similarity = user_ratings.T.corr()

# Content-based TF-IDF similarity
tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[\w\-]+\b')
tfidf_matrix = tfidf.fit_transform(movies_meta['text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_meta.index, index=movies_meta['title']).drop_duplicates()

# Item-based similarity matrix
item_corr = user_movie_matrix.corr(method='pearson', min_periods=50)


# ---------- STREAMLIT UI ----------
st.title("🎬 Movie Recommendation System")
st.markdown("Choose your recommendation strategy:")

option = st.radio(
    "Select filtering method:",
    ('Content-Based (Title + Genre)', 'Item-Based Collaborative Filtering', 'User-Based Collaborative Filtering'))

if option == 'Content-Based (Title + Genre)':
    movie = st.selectbox("Choose a movie:", sorted(movies_meta['title'].unique()))
    n = st.slider("Number of recommendations", 3, 10, 5)

    def recommend_cb(title, n=5):
        if title not in indices:
            return []
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_meta['title'].iloc[movie_indices].tolist()

    if st.button("Recommend"):
        results = recommend_cb(movie, n)
        st.write("Recommended movies:")
        for r in results:
            st.markdown(f"• {r}")

elif option == 'Item-Based Collaborative Filtering':
    movie = st.selectbox("Choose a movie:", sorted(user_movie_matrix.columns))
    n = st.slider("Number of recommendations", 3, 10, 5)

    def recommend_item_based(movie, n=5):
        if movie not in item_corr.columns:
            return []
        similar = item_corr[movie].dropna().sort_values(ascending=False)
        return similar.drop(labels=movie).head(n).index.tolist()

    if st.button("Recommend"):
        results = recommend_item_based(movie, n)
        st.write("Movies rated similarly by users:")
        for r in results:
            st.markdown(f"• {r}")

elif option == 'User-Based Collaborative Filtering':
    user_id = st.number_input("Enter your user ID (1–943)", min_value=1, max_value=943, step=1)
    n = st.slider("Number of recommendations", 3, 10, 5)

    def predict_ratings_for_user(user_id, n=5, min_sim=0.1):
        if user_id not in user_ratings.index:
            return []
        rated = user_ratings.loc[user_id].dropna().index
        sim_scores = user_similarity[user_id].drop(index=user_id)
        sim_scores = sim_scores[sim_scores > min_sim]
        weighted_sum = {}
        sim_sum = {}

        for other, sim in sim_scores.items():
            other_rated = user_ratings.loc[other].dropna()
            for movie, rating in other_rated.items():
                if movie not in rated:
                    weighted_sum[movie] = weighted_sum.get(movie, 0) + rating * sim
                    sim_sum[movie] = sim_sum.get(movie, 0) + abs(sim)

        preds = pd.Series({m: weighted_sum[m] / sim_sum[m] for m in weighted_sum if sim_sum[m] != 0})
        return preds.sort_values(ascending=False).head(n).index.tolist()

    if st.button("Recommend"):
        results = predict_ratings_for_user(user_id, n)
        st.write(f"Recommended for user {user_id}:")
        for r in results:
            st.markdown(f"• {r}")
