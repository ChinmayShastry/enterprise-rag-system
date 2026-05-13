"""
rag/generation.py
LLM answer generation with streaming support.
The assistant persona and document context are fully configurable.
"""

from openai import OpenAI
from langchain_core.documents import Document


def get_sources(docs: list[Document]) -> list[str]:
    """
    Extract unique, sorted page citations from a list of documents.
    Only the docs actually passed to the LLM are listed as sources.
    """
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page")
        if page is not None:
            pages.add(f"Page {page}")
    return sorted(pages, key=lambda s: int(s.split()[-1]))


def _build_context(docs: list[Document]) -> str:
    context = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        context += f"[Chunk {i + 1} — Page {page}]\n{doc.page_content}\n\n"
    return context


def _build_prompt(question: str, context: str, persona: str) -> str:
    return (
        f"You are {persona}.\n"
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not present in the context, say so clearly — do not guess.\n"
        "Always cite which page your answer comes from.\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def stream_answer(
    client: OpenAI,
    question: str,
    docs: list[Document],
    llm_model: str,
    persona: str,
):
    """
    Generator that yields answer tokens one by one for st.write_stream().
    Using streaming gives immediate feedback instead of a blank screen.
    """
    context = _build_context(docs)
    prompt = _build_prompt(question, context, persona)

    stream = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
