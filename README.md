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

Install the Git hooks once in each clone:

```bash
pre-commit install
```

## Code quality

Ruff provides linting, import sorting, and formatting. Coverage measures both
statements and branches across the package, command-line interface, and
Streamlit application. The test suite fails below 95% total coverage.

Run the same checks used by continuous integration with:

```bash
ruff check .
ruff format --check app.py run_rag.py rag_pipeline tests
pytest
pre-commit run --all-files
```

To apply Ruff's safe fixes and formatting locally, run:

```bash
ruff check --fix .
ruff format app.py run_rag.py rag_pipeline tests
```

Coverage writes an interactive report to `htmlcov/index.html` and a
machine-readable report to `coverage.xml`. Ruff lints `rag_test.ipynb`, while
the formatter deliberately leaves the notebook unchanged.

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
