import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import PyPDF2
from io import BytesIO
from transformers import BertTokenizer, BertModel
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

#----------------------------------------------------------------------------------------------------------#
# Initialize BERT model and tokenizer
# Bidirectional Encoder Representations from Transformers BERT
@st.cache_resource
def load_bert_model():
    """Load BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model
#----------------------------------------------------------------------------------------------------------#

# Summarize text using Latent Semantic Analysis (LSA)

def lsa_summarize(text, num_sentences=3):

    if not text.strip():
        return "Please enter some text to summarize."

    # Clean and tokenize into sentences
    sentences = sent_tokenize(text)
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]

    if len(sentences) == 0:
        return "Text is too short to summarize."

    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    try:
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Apply SVD (LSA)
        n_components = min(min(10, len(sentences) - 1), tfidf_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components) # returns counts/tf-idf value
        svd_matrix = svd.fit_transform(tfidf_matrix) # fit the model and perform dim reduction

        # Calculate sentence scores based on singular values
        sentence_scores = np.sqrt(np.sum(svd_matrix ** 2, axis=1))

        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

#----------------------------------------------------------------------------------------------------------#
 # Summarize text using TextRank Algorithm
def textrank_summarize(text, num_sentences=3):

    if not text.strip():
        return "Please enter some text to summarize."

    # Clean and tokenize into sentences
    sentences = sent_tokenize(text)
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]

    if len(sentences) == 0:
        return "Text is too short to summarize."

    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    try:
        # Create TF-IDF matrix for similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

        # Add small value to avoid isolated nodes
        similarity_matrix = similarity_matrix + 0.01

        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=100)

        # Get top sentences
        ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_indices = sorted([idx for _, idx in ranked_sentences[:num_sentences]])

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ----------------------------------------------------------------------------------------------------------#
    # Summarize text using BERT embeddings
# Bidirectional Encoder Representations from Transformers BERT
def bert_summarize(text, num_sentences=3):

    if not text.strip():
        return "Please enter some text to summarize."

    # Clean and tokenize into sentences
    sentences = sent_tokenize(text)
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]

    if len(sentences) == 0:
        return "Text is too short to summarize."

    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    try:
        tokenizer, model = load_bert_model()

        # Get BERT embeddings for each sentence
        sentence_embeddings = []
        for sentence in sentences:
            # Tokenize and get model output
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use CLS token embedding as sentence representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            sentence_embeddings.append(sentence_embedding)

        sentence_embeddings = np.array(sentence_embeddings)

        # Calculate document embedding (mean of all sentences)
        doc_embedding = np.mean(sentence_embeddings, axis=0)

        # Calculate similarity between each sentence and document
        similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]

        # Get top sentences
        top_indices = similarities.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

#----------------------------------------------------------------------------------------------------------#
#Extract texxt from files
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def extract_text_from_txt(txt_file):
    """Extract text from uploaded TXT file"""
    try:
        # Try different encodings
        try:
            text = txt_file.read().decode('utf-8')
        except UnicodeDecodeError:
            txt_file.seek(0)
            text = txt_file.read().decode('latin-1')
        return text
    except Exception as e:
        return f"Error reading TXT file: {str(e)}"

#----------------------------------------------------------------------------------------------------------#
# Streamlit UI start

st.set_page_config(page_title="Smart Notepad", layout="wide")

# Handle clear flag
if 'clear_flag' in st.session_state and st.session_state['clear_flag']:
    st.session_state['notepad'] = ""
    st.session_state['clear_flag'] = False

st.title("📝 Notepad with Text Summarization")
st.markdown("Write or upload your text and get summary using LSA or TextRank or BERT algorithms.")

# Settings sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    algorithm = st.selectbox(
        "Summarization Algorithm",
        ["LSA (Latent Semantic Analysis)", "TextRank", "BERT (Transformer-based)"]
    )
    num_sentences = st.slider(
        "Number of sentences in summary",
        min_value=2,
        max_value=15,
        value=5,
        help="Increase this for longer summaries"
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **LSA**: Uses singular value decomposition to identify key semantic concepts. Best for research papers.

    **TextRank**: Graph-based ranking algorithm inspired by PageRank. Best for news articles.

    **BERT**: Uses transformer-based deep learning embeddings for semantic understanding. Best for complex text files.

    **File Support**: Upload and extract text from PDF and TXT documents.
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Your Notes")

    # File Upload section
    with st.expander("📎 Upload Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a PDF or TXT file to extract and summarize its content"
        )

        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            button_label = f"📖 Read & Load {file_type.upper()}"

            if st.button(button_label, use_container_width=True):
                with st.spinner(f"Extracting text from {file_type.upper()}..."):
                    if file_type == 'pdf':
                        extracted_text = extract_text_from_pdf(uploaded_file)
                    else:  # txt
                        extracted_text = extract_text_from_txt(uploaded_file)

                    if extracted_text.startswith("Error"):
                        st.error(extracted_text)
                    else:
                        st.session_state['notepad'] = extracted_text
                        word_count_file = len(extracted_text.split())
                        st.success(f"{file_type.upper()} loaded successfully! ({word_count_file} words)")
                        st.rerun()

    text_input = st.text_area(
        "Write or paste your text here:",
        height=350,
        placeholder="Start typing, paste your text, or upload a PDF/TXT file...",
        key="notepad"
    )

    word_count = len(text_input.split())
    char_count = len(text_input)
    st.caption(f"Words: {word_count} | Characters: {char_count}")

with col2:
    st.subheader("Summary")

    if st.button("Generate Summary", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Generating summary..."):
                if "LSA" in algorithm:
                    summary = lsa_summarize(text_input, num_sentences)
                elif "TextRank" in algorithm:
                    summary = textrank_summarize(text_input, num_sentences)
                else:  # BERT
                    summary = bert_summarize(text_input, num_sentences)

                st.session_state['summary'] = summary
        else:
            st.warning("Please enter some text first!")

    if 'summary' in st.session_state:
        st.text_area(
            "Generated Summary:",
            value=st.session_state['summary'],
            height=400,
            key="summary_output"
        )

        summary_words = len(st.session_state['summary'].split())
        if word_count > 0:
            reduction = round((1 - summary_words / word_count) * 100, 1)
            st.caption(f"Summary words: {summary_words} | Reduction: {reduction}%")

# Action buttons
st.markdown("---")
col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    if st.button("Clear All", use_container_width=True):
        # Use a flag to clear on next rerun
        st.session_state['clear_flag'] = True
        if 'summary' in st.session_state:
            del st.session_state['summary']
        st.rerun()

with col_b:
    if 'summary' in st.session_state and st.download_button(
            label="Download Summary",
            data=st.session_state['summary'],
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True
    ):
        st.success("Summary downloaded!")