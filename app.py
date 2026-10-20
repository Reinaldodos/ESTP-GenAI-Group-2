"""Interface Streamlit du pipeline RAG modulaire."""

from dataclasses import replace
from pathlib import Path

import streamlit as st

from rag_pipeline import RAGConfig, RAGPipeline

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@st.cache_resource
def initialize_rag(
    config: RAGConfig, chunk_size: int, chunk_overlap: int
) -> RAGPipeline:
    """Construit et met en cache l'index pour les paramètres de découpage."""
    runtime_config = replace(
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return RAGPipeline(runtime_config)


def main() -> None:
    st.set_page_config(page_title="Travel RAG", page_icon="✈️", layout="wide")
    st.title("✈️ Travel Policy Assistant")
    st.caption("RAG demo using BGE embeddings, FAISS, and a local LLM")

    config = RAGConfig.from_yaml(CONFIG_PATH)

    with st.sidebar:
        st.header("RAG Settings")
        chunk_size = st.slider(
            "Chunk size",
            min_value=100,
            max_value=2000,
            value=config.chunk_size,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=chunk_size - 50,
            value=min(config.chunk_overlap, chunk_size - 50),
            step=50,
        )
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=config.retrieval_k,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config.llm_temperature,
            step=0.1,
        )

    try:
        pipeline = initialize_rag(config, chunk_size, chunk_overlap)
    except Exception as error:
        st.error("The RAG index could not be initialized.")
        st.exception(error)
        st.stop()

    with st.sidebar:
        st.divider()
        st.write(f"**Embedding:** `{config.embedding_model}`")
        st.write(f"**LLM:** `{config.llm_model}`")
        st.write(f"**Documents:** {len(pipeline.document_sources)}")
        st.write(f"**Chunks:** {pipeline.chunk_count}")
        st.divider()
        st.subheader("Documents")
        for source in pipeline.document_sources:
            st.write(f"📄 {source}")

    question = st.text_input(
        "Ask a question about business travel",
        placeholder="e.g. Which hotels can I book without manager approval?",
    )

    if not question:
        st.info("Ask a question to search the travel documents.")
        st.subheader("Try these questions")
        for example in (
            "How much can a hotel cost?",
            "What is the maximum meal allowance?",
            "Which hotels are available in Amsterdam?",
            "Which hotels can I book without manager approval?",
            "Can I travel in business class?",
            "Does the Canal View Hotel have a swimming pool?",
        ):
            st.markdown(f"- {example}")
        return

    with st.spinner("Searching documents..."):
        try:
            response = pipeline.ask(
                question,
                k=top_k,
                temperature=temperature,
            )
        except Exception as error:
            st.error(
                f"The request failed. Make sure LM Studio is running and "
                f"the `{config.llm_model}` model is loaded."
            )
            st.exception(error)
            st.stop()

    st.subheader("Answer")
    st.write(response.answer)
    st.subheader("Retrieved Documents")
    for position, result in enumerate(response.sources, start=1):
        label = f"{position}. {result['source']} — similarity {result['score']:.3f}"
        with st.expander(label):
            st.write(f"**Chunk:** {result['chunk_id']}")
            st.write(result["text"])


main()
