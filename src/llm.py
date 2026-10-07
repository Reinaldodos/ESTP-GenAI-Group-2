from openai import OpenAI


def create_llm_client(base_url, api_key):
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def generate_answer(
    question,
    results,
    client,
    model,
    temperature=0,
):
    context = "\n\n".join(
        f"[Source: {result['source']}]\n"
        f"{result['text']}"
        for result in results
    )

    prompt = f"""
Answer the user's question using ONLY the information
contained in the context below.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Do not invent information.

Context:
----------------
{context}
----------------

Question:
{question}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content
