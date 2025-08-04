import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets with error handling
@st.cache_data
def load_books():
    df = pd.read_csv("books.csv", encoding='ISO-8859-1', delimiter=';', on_bad_lines='skip', low_memory=False)
    df.rename(columns={"Book-Title": "title", "Book-Author": "author"}, inplace=True)
    df = df.dropna(subset=["title"])
    df["content"] = df["title"] + " " + df["author"].fillna("")
    return df

@st.cache_data
def load_ratings():
    return pd.read_csv("ratings.csv", encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

@st.cache_data
def load_users():
    return pd.read_csv("users.csv", encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

@st.cache_resource
def compute_tfidf(contents):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(contents)
    return tfidf_matrix

def recommend_books(query, books_df, tfidf_matrix):
    query_vec = TfidfVectorizer(stop_words="english").fit(books_df["content"]).transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    return books_df.iloc[top_indices]

# --- UI Styling ---
st.set_page_config(page_title="📚 Book Recommender", layout="centered", page_icon="📘")

st.markdown("""
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 30px;
    }
    .recommend-box {
        background-color: #f5faff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .recommend-title {
        font-size: 20px;
        font-weight: bold;
        color: #1a1a1a;
    }
    .recommend-author {
        font-size: 16px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📚 Book Recommender</div>', unsafe_allow_html=True)

books_df = load_books()
ratings_df = load_ratings()
users_df = load_users()

if not books_df.empty:
    tfidf_matrix = compute_tfidf(books_df["content"])

    user_query = st.text_input("🔍 Type a title, author or keyword to get recommendations:")

    if user_query:
        recommendations = recommend_books(user_query, books_df, tfidf_matrix)
        st.markdown("### 🎯 Recommended Reads:")
        for _, row in recommendations.iterrows():
            st.markdown(f"""
                <div class="recommend-box">
                    <div class="recommend-title">📖 {row['title']}</div>
                    <div class="recommend-author">✍️ {row['author']}</div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.error("❌ Unable to load book data. Please check your CSV files.")
