import pickle
import streamlit as st
import numpy as np
import base64

# Set page configuration
st.set_page_config(page_title="Book Recommender System", layout="wide")

# Function to encode image to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode the background image
img_base64 = get_base64("Dall-E.webp")

# Add custom CSS for background image and styling
page_bg_img = f'''
<style>
body {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    color: white;
}}
.sidebar .sidebar-content {{
    background: rgba(0, 0, 0, 0.5);
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add a header and subheader
st.title('Capstone Project: Book Recommender System - SIC')
st.subheader('Find your next favorite book!')

# Add descriptive text
st.markdown("""
Welcome to the Book Recommender System. Select a book you like, and we will recommend similar books for you.
""")

# Load the model and data
model = pickle.load(open('artifacts/model.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))

# Function to fetch poster URLs
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

# Function to recommend books
def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]

    distance, suggestion = model.kneighbors(
        book_pivot.iloc[book_id,:].values.reshape(1,-1),
        n_neighbors=6 
    )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

# Function to store recommended books
def store_recommended_books(book_name, recommended_books):
    with open('recommended_books.txt', 'a') as f:
        f.write(f"{book_name}: {', '.join(recommended_books)}\n")

# Add a selectbox for book selection
selected_books = st.selectbox(
    "Select a book you like:",
    book_names[1:]
)

# Add a button to get recommendations
if st.button('Get Recommendations'):
    recommended_books, poster_url = recommend_book(selected_books)
    store_recommended_books(selected_books, recommended_books)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(recommended_books[i+1])
            st.image(poster_url[i+1])

# Display previously recommended books
st.subheader('Previously Recommended Books')
try:
    with open('recommended_books.txt', 'r') as f:
        st.text(f.read())
except FileNotFoundError:
    st.text("No recommendations yet.")