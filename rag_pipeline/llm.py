"""Construction du prompt et appel au modèle servi par LM Studio."""

from openai import OpenAI

from .retrieval import SearchResult


FALLBACK_ANSWER = "I don't know based on the provided documents."


def create_client(base_url: str, api_key: str) -> OpenAI:
    """Crée un client compatible avec l'API OpenAI de LM Studio."""
    return OpenAI(base_url=base_url, api_key=api_key)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Assemble les passages retrouvés dans un prompt contraint."""
    context = "\n\n".join(
        f"[Source: {result['source']}]\n{result['text']}" for result in results
    )
    return f"""Answer the user's question using ONLY the information
contained in the context below.

If the answer cannot be found in the context, say:
\"{FALLBACK_ANSWER}\"

Do not invent information.

Context:
----------------
{context}
----------------

Question:
{question}
"""


def generate_answer(
    client: OpenAI, model: str, question: str, results: list[SearchResult]
) -> str:
    """Envoie au LLM la question enrichie par les résultats de recherche."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
        temperature=0,
    )
    return response.choices[0].message.content or ""
