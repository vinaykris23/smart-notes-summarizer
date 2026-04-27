# Smart Notepad with Text Summarization

A Streamlit-based web application that summarizes text using three NLP algorithms: **LSA (Latent Semantic Analysis)**, **TextRank**, and **BERT (Transformer-based)**.

---

## Features

- **Three Summarization Algorithms:**
  - **LSA** — Uses Singular Value Decomposition (SVD) to identify key semantic concepts. Best for research papers.
  - **TextRank** — Graph-based ranking algorithm inspired by PageRank. Best for news articles.
  - **BERT** — Uses transformer-based deep learning embeddings for semantic understanding. Best for complex texts.
- **File Upload Support** — Upload and extract text from PDF and TXT documents.
- **Adjustable Summary Length** — Choose how many sentences to include in the summary.
- **Download Summary** — Save the generated summary as a `.txt` file.
- **Word & Character Count** — Live stats shown as you type or load a file.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-notepad-summarizer.git
cd smart-notepad-summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run v14b.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Requirements

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `numpy` | Numerical computations |
| `scikit-learn` | TF-IDF vectorizer & SVD (LSA) |
| `networkx` | Graph construction for TextRank |
| `nltk` | Sentence tokenization |
| `PyPDF2` | PDF text extraction |
| `transformers` | BERT model (Hugging Face) |
| `torch` | PyTorch backend for BERT |

---

## Project Structure

```
smart-notepad-summarizer/
│
├── v14b.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── sample_text_dataset_.txt   # Sample text for testing
```

---

## Usage

1. **Type or paste** text into the notepad area, **or**
2. **Upload** a `.pdf` or `.txt` file using the Upload Document panel
3. Select a **Summarization Algorithm** from the sidebar
4. Adjust the **number of sentences** using the slider
5. Click **Generate Summary**
6. Optionally **Download** the summary as a text file

---

## Deploy on Streamlit Cloud (Free)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set **Main file path** to `v14b.py`
5. Click **Deploy** — you'll get a shareable public URL!

---

## Research Background

This project is based on the following NLP techniques:

- **LSA**: Deerwester et al. (1990) — Latent Semantic Analysis using SVD on TF-IDF matrices
- **TextRank**: Mihalcea & Tarau (2004) — Graph-based unsupervised extractive summarization
- **BERT**: Devlin et al. (2018) — Bidirectional Encoder Representations from Transformers

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

Built as part of an NLP research project on text summarization techniques.
