from pathlib import Path


def load_documents(directory, pattern="*.md"):
    documents = []

    for path in Path(directory).glob(pattern):
        text = path.read_text(encoding="utf-8")

        documents.append({
            "source": path.name,
            "text": text,
        })

    return documents
