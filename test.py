import streamlit as st
import os
import string
import math
from streamlit_navigation_bar import st_navbar
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fitz


def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)
    magnitude1 = math.sqrt(sum(vector1[key] ** 2 for key in vector1))
    magnitude2 = math.sqrt(sum(vector2[key] ** 2 for key in vector2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def calculate_similarity(doc1, doc2, n):
    doc1_ngrams = Counter(get_ngrams(doc1, n))
    doc2_ngrams = Counter(get_ngrams(doc2, n))
    return cosine_similarity(doc1_ngrams, doc2_ngrams)

def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def load_documents(folder):
    documents = {}
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
            documents[filename] = preprocess_text(file.read())
    return documents

def train_test_model(original_text, suspicious_text, n):
    original_documents = {'original.txt': preprocess_text(original_text)}
    suspicious_documents = {'suspicious.txt': preprocess_text(suspicious_text)}

    # Prepare data
    X = []
    y = []
    for suspicious_filename, suspicious_content in suspicious_documents.items():
        for original_filename, original_content in original_documents.items():
            similarity = calculate_similarity(suspicious_content, original_content, n)
            X.append((suspicious_content, original_content))
            # Label 1 if similarity is above threshold, 0 otherwise
            y.append(1 if similarity > 0.8 else 0)

    # Dummy classifier that predicts no plagiarism (0) for all cases
    y_pred_test = [0] * len(y)

    # Calculate accuracy
    test_accuracy = accuracy_score(y, y_pred_test)

    return test_accuracy, similarity

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["DOC/TXT", "Raw Text"]
styles = {   
    "nav": {
        "background-color": "rgb(58,70,100)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255,255,255)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    }
}

page = st_navbar(pages, styles=styles)
st.write(page)

if page == "DOC/TXT":

    st.title("Plagiarism Detection System")
    st.write("Upload an original document and a suspicious document to check for plagiarism.")

    original_file = st.file_uploader("Upload Original Document", type=["txt", "pdf"])
    suspicious_file = st.file_uploader("Upload Suspicious Document", type=["txt", "pdf"])

    n = st.slider("Select n-gram size", 1, 10, 2)

    if original_file is not None and suspicious_file is not None:
        if original_file.type == "application/pdf":
            original_text = extract_text_from_pdf(original_file)
        else:
            original_text = original_file.read().decode("utf-8")

        if suspicious_file.type == "application/pdf":
            suspicious_text = extract_text_from_pdf(suspicious_file)
        else:
            suspicious_text = suspicious_file.read().decode("utf-8")

        test_accuracy, similarity = train_test_model(original_text, suspicious_text, n)

        st.write(f"Test Accuracy: {test_accuracy:.2f}")
        st.write(f"Similarity: {similarity:.2f}")
        if similarity > 0.8:
            st.write("The documents are likely plagiarized.")
        else:
            st.write("The documents are not plagiarized.")

elif page == "Raw Text":
    
    class n_gram:
        def __init__(self, n=2):
            self.n = n

        def preprocess_text(self, text):
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            text = text.lower()  # Convert to lowercase
            return text
        def get_ngrams(self, text, n):
            tokens = text.split()
            ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            return ngrams
        def calculate(self, text1, text2):
            """
            Preprocess text and Turn text into ngrams
            """
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)

            ngrams_text1 = Counter(self.get_ngrams(text1, self.n))
            ngrams_text2 = Counter(self.get_ngrams(text2, self.n))

            """
            Calculates the cosine similarity between two vectors.
            """
            dot_product = sum(ngrams_text1[key] * ngrams_text2.get(key, 0) for key in ngrams_text1)
            magnitude1 = math.sqrt(sum(ngrams_text1[key] ** 2 for key in ngrams_text1))
            magnitude2 = math.sqrt(sum(ngrams_text2[key] ** 2 for key in ngrams_text2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            return round(dot_product / (magnitude1 * magnitude2), 2)
        

        def get_x(self, text1, text2):

            """
            Preprocess text and Turn text into ngrams
            """
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)

            ngrams_text1 = Counter(self.get_ngrams(text1, self.n))
            ngrams_text2 = Counter(self.get_ngrams(text2, self.n))
            
            return ngrams_text1, ngrams_text2
        







    ## find the similarity between 2 text using ngram and cosine similarity
    # Streamlit interface
    st.title("N-Gram Cosine Similarity")

    st.write("Enter two texts to compute their cosine similarity based on n-gram vectors.")

    text1 = st.text_area("Text 1", "")
    text2 = st.text_area("Text 2", "")
    n = st.slider("Select n-gram size", 1, 10, 2)

    if st.button("Compute Similarity"):
        if text1 and text2 and n:
            model = n_gram(n)
            similarity_score = model.calculate(text1, text2)
            st.write(f"Cosine Similarity (based on {n}-gram): {similarity_score:.4f}")
        else:
            st.write("Please enter text in both fields and specify n-gram.")

    