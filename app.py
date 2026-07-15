import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
from openai import OpenAI
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=st.secrets["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"]
)
import tempfile
from nltk.tokenize import sent_tokenize
import re
import nltk
nltk.download('punkt_tab')
# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI PDF Assistant"
)

st.title("AI PDF Assistant with Citations")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# CHAT MEMORY
# -----------------------------------
if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------
@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_embedding_model()
def rerank(query, chunks, top_k=5):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
        overlap = len(query_words & chunk_words) / (len(query_words) + 1)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
def rewrite_query(original_query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': f"""Rewrite this question to be more specific and detailed 
            for searching a PDF document. Return ONLY the rewritten question, nothing else.
            Original question: {original_query}"""}]
    )
    return response.choices[0].message.content.strip()

# -----------------------------------
# PROCESS PDF
# -----------------------------------
@st.cache_resource
def process_pdf(pdf_bytes):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(pdf_bytes)

        pdf_path = tmp_file.name

    # -----------------------------------
    # READ PDF
    # -----------------------------------
    reader = PdfReader(pdf_path)

    # -----------------------------------
    # SMART CHUNKING WITH PAGE METADATA
    # -----------------------------------
    chunks = []

    chunk_size = 5
    overlap = 3

    for page_num, page in enumerate(reader.pages):

        extracted = page.extract_text()

        if extracted:

            sentences = sent_tokenize(
                extracted
            )

            for i in range(
                0,
                len(sentences),
                chunk_size - overlap
            ):

                chunk = sentences[
                    i:i + chunk_size
                ]

                chunk = " ".join(chunk)

                chunks.append(
                    {
                        "text": chunk,
                        "page": page_num + 1
                    }
                )

    # -----------------------------------
    # EXTRACT TEXTS FOR EMBEDDINGS
    # -----------------------------------
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------
    # GENERATE EMBEDDINGS
    # -----------------------------------
    embeddings = model.encode(texts)

    embeddings = np.array(
        embeddings
    ).astype('float32')

    # -----------------------------------
    # CREATE FAISS INDEX
    # -----------------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return chunks, index

# -----------------------------------
# FILE UPLOADER
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------------
# MAIN APP
# -----------------------------------
if uploaded_file:
    with st.spinner("Processing PDFs..."):
        all_chunks = []
        all_embeddings = []
        for file in uploaded_file:
            pdf_bytes = file.read()
            chunks, index = process_pdf(pdf_bytes)
            all_chunks.extend(chunks)

        chunks = all_chunks
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

    st.success(
        "PDF processed successfully!"
    )

    # -----------------------------------
    # USER QUERY
    # -----------------------------------
    query = st.text_input(
        "Ask a question about the PDF"
    )

    # -----------------------------------
    # ASK BUTTON
    # -----------------------------------
    if st.button("Ask"):

        if query:

            # Save user message
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": query
                }
            )

            with st.spinner(
                "Generating answer..."
            ):

                # -----------------------------------
                # QUERY EMBEDDING
                # -----------------------------------
                search_query = rewrite_query(query)
                query_embedding = model.encode([search_query])

                query_embedding = np.array(
                    query_embedding
                ).astype('float32')

                # -----------------------------------
                # SEARCH VECTOR DATABASE
                # -----------------------------------
                k = 8

                distances, indices = index.search(
                    query_embedding,
                    k
                )

                # -----------------------------------
                # RETRIEVE RELEVANT CHUNKS
                # -----------------------------------
                retrieved_chunks = []

                for idx in indices[0]:

                    retrieved_chunks.append(
                        chunks[idx]
                    )
                retrieved_chunks = rerank(query, retrieved_chunks, top_k=5)

                # -----------------------------------
                # BUILD CONTEXT
                # -----------------------------------
                context = "\n".join(
                    [
                        chunk["text"]
                        for chunk in retrieved_chunks
                    ]
                )

                # -----------------------------------
                # BUILD CONVERSATION HISTORY
                # -----------------------------------
                conversation_history = ""

                for message in st.session_state.chat_history:

                    role = message["role"]

                    content = message["content"]

                    conversation_history += (
                        f"{role}: {content}\n"
                    )

                # -----------------------------------
                # CREATE PROMPT
                # -----------------------------------
                prompt = f"""
                You are an AI PDF assistant.

                Use the conversation history
                and PDF context below
                to answer the user's question.

                Conversation History:
                {conversation_history}

                Context:
                {context}

                Question:
                {query}
                """

                try:

                    # -----------------------------------
                    # GENERATE RESPONSE
                    # -----------------------------------
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{'role': 'user', 'content': prompt}]
                    )
                    answer = response.choices[0].message.content

                    # Save assistant response
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    # -----------------------------------
                    # DISPLAY ANSWER
                    # -----------------------------------
                    with st.chat_message("assistant"):
                        st.write(answer)

                    # -----------------------------------
                    # DISPLAY SOURCES
                    # -----------------------------------
                    with st.expander("📚 Sources"):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.caption(f"**Source {i+1} — Page {chunk['page']}**")
                            st.caption(chunk["text"])
                            st.divider()

                except Exception as e:

                    st.error(f"Error: {e}")

    # -----------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
