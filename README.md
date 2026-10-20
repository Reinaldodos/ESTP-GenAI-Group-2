# ESTP Travel RAG

This project provides a modular retrieval-augmented generation pipeline for the
included corporate travel documents. It can be used from the command line or
through a Streamlit interface. The configured language model is served locally
by LM Studio.

## Installation

Create and activate a Python 3.10+ virtual environment, then install the project
with the interface dependencies:

```bash
python -m pip install -e '.[app]'
```

For development and tests, install both optional dependency groups:

```bash
python -m pip install -e '.[app,test]'
```

## Streamlit application

Start LM Studio, load the model named in `config.yaml`, and expose its
OpenAI-compatible server. Then run:

```bash
streamlit run app.py
```

The sidebar controls chunk size, chunk overlap, retrieved passage count, and
generation temperature. Changing the chunk settings rebuilds the vector index;
changing the passage count or temperature reuses the existing index.

## Command-line interface

With LM Studio running, launch the interactive CLI with:

```bash
python run_rag.py
```
