import streamlit as st
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer


st.title("Text Summarization Generation")

text = st.text_area("Paste or write your text")

number_of_summary = st.number_input(
    "Number of sentence for summary",
    min_value=1, max_value=15, value=5, step=1
)

if st.button("Summarize"):
    if text:
        with st.spinner("Generating Summary"):
            sentences = sent_tokenize(text)

            # Make all text lowercase
            cleaned_sentences = [s.lower() for s in sentences]

            stop_words = stopwords.words('english')
            def remove_stop_words(sent):
                return ' '.join([i for i in sent.split() if i not in stop_words])
            
            cleaned_senetences_no_stopwords = [remove_stop_words(i) for i in cleaned_sentences]

            @st.cache_resource
            def load_embedding():
                return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            model = load_embedding()

            sentences_vectors = model.encode(cleaned_senetences_no_stopwords) # Vectorization sentences 

            similarity_matrix = np.zeros([len(sentences), len(sentences)])

            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        similarity_matrix[i][j] = cosine_similarity(
                            sentences_vectors[i].reshape(1,-1),
                            sentences_vectors[j].reshape(1,-1),
                        )[0,0]

            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)

            ranked_sentences = sorted(((scores[i], s, i) for i,s in enumerate(sentences)), reverse=True)

            summary_sentences = sorted(ranked_sentences[:number_of_summary], key=lambda x:x[2])

            st.subheader("summary:")
            summary_text = ""
            for _,sentence,_ in summary_sentences:
                summary_text += sentence

            st.write(summary_text)