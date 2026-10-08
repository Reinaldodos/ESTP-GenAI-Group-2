from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from src.config import load_config
from src.document_loader import load_documents
from src.chunking import create_chunks
from src.embedding import create_embedding_model, create_embeddings
from src.retrieval import create_faiss_index, retrieve
from src.llm import create_llm_client, generate_answer


# =========================================================
# Environment
# =========================================================

load_dotenv()


# =========================================================
# Configuration
# =========================================================

config = load_config("config.yaml")

documents_directory = Path(
    config["documents"]["directory"]
)

documents_pattern = config["documents"]["file_pattern"]

embedding_model_name = config["embedding"]["model"]

top_k = config["retrieval"]["top_k"]

llm_model_name = config["llm"]["model"]

llm_temperature = config["llm"]["temperature"]

chunk_size = config["chunking"]["chunk_size"]

chunk_overlap = config["chunking"]["overlap"]

base_url = os.getenv("LLM_BASE_URL")

api_key = os.getenv("LLM_API_KEY")


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Travel RAG",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Travel Policy Assistant")

st.caption(
    "RAG demo using BGE embeddings, FAISS, and Qwen3-8B"
)

# =========================================================
# Sidebar II
# =========================================================

with st.sidebar:

    st.header("RAG Settings")

    chunk_size_selected = st.slider(
        "Chunk size",
        min_value=100,
        max_value=2000,
        value=chunk_size,
        step=100,
    )

    chunk_overlap_selected = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=chunk_size_selected,
        value=min(chunk_overlap, chunk_size_selected),
        step=50,
    )
    
    top_k_selected = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=top_k,
    )

    temperature_selected = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(llm_temperature),
        step=0.1,
    )


# =========================================================
# Initialize RAG
# =========================================================

@st.cache_resource
def initialize_rag():

    # -----------------------------------------------------
    # 1. Load documents
    # -----------------------------------------------------

    documents = load_documents(
        documents_directory,
        documents_pattern,
    )

    # -----------------------------------------------------
    # 2. Create chunks
    # -----------------------------------------------------

    chunks = create_chunks(
        documents,
        chunk_size=chunk_size_selected,
        overlap=chunk_overlap_selected,
    )

    # -----------------------------------------------------
    # 3. Load embedding model
    # -----------------------------------------------------

    embedding_model = create_embedding_model(
        embedding_model_name
    )

    # -----------------------------------------------------
    # 4. Create document embeddings
    # -----------------------------------------------------

    embeddings = create_embeddings(embedding_model, chunks)

    # -----------------------------------------------------
    # 5. Create FAISS index
    # -----------------------------------------------------

    index = create_faiss_index(
        embeddings
    )

    # -----------------------------------------------------
    # 6. Create LLM client
    # -----------------------------------------------------

    llm_client = create_llm_client(
        base_url=base_url,
        api_key=api_key,
    )

    return {
        "documents": documents,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "index": index,
        "llm_client": llm_client,
    }


rag = initialize_rag()


# =========================================================
# Sidebar II
# =========================================================

with st.sidebar:
    st.divider()

    st.write(
        f"**Embedding:** "
        f"`{embedding_model_name}`"
    )

    st.write(
        f"**LLM:** "
        f"`{llm_model_name}`"
    )

    st.write(
        f"**Documents:** "
        f"{len(rag['documents'])}"
    )

    st.write(
        f"**Chunks:** "
        f"{len(rag['chunks'])}"
    )

    st.divider()
    
    st.subheader("Documents")
    
    for document in rag["documents"]:

        st.write(
            f"📄 {document['source']}"
        )


# =========================================================
# Main interface
# =========================================================

question = st.text_input(
    "Ask a question about business travel",
    placeholder=(
        "e.g. Which hotels can I book "
        "without manager approval?"
    ),
)


# =========================================================
# Process question
# =========================================================

if question:

    with st.spinner("Searching documents..."):

        try:

            # -------------------------------------------------
            # Retrieve relevant chunks
            # -------------------------------------------------

            results = retrieve(
                question=question,
                embedding_model=rag["embedding_model"],
                index=rag["index"],
                chunks=rag["chunks"],
                top_k=top_k_selected,
            )

            # -------------------------------------------------
            # Generate answer
            # -------------------------------------------------

            answer = generate_answer(
                question=question,
                results=results,
                client=rag["llm_client"],
                model=llm_model_name,
                temperature=temperature_selected,
            )

        except Exception as e:

            st.error(
                "Something went wrong. "
                "Make sure LM Studio is running "
                "and the Qwen model is loaded."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # Answer
    # =====================================================

    st.subheader("Answer")

    st.write(answer)


    # =====================================================
    # Retrieved documents
    # =====================================================

    st.subheader("Retrieved Documents")

    for i, result in enumerate(
        results,
        start=1,
    ):

        with st.expander(
            f"{i}. {result['source']} "
            f"— similarity "
            f"{result['score']:.3f}"
        ):

            st.write(
                f"**Chunk:** "
                f"{result['chunk_id']}"
            )

            st.write(
                result["text"]
            )


# =========================================================
# Example questions
# =========================================================

else:

    st.info(
        "Ask a question to search "
        "the travel documents."
    )

    st.subheader(
        "Try these questions"
    )

    example_questions = [
        "How much can a hotel cost?",
        "What is the maximum meal allowance?",
        "Which hotels are available in Amsterdam?",
        "Which hotels can I book without manager approval?",
        "Can I travel in business class?",
        "Does the Canal View Hotel have a swimming pool?",
    ]

    for example in example_questions:

        st.markdown(
            f"- {example}"
        )
