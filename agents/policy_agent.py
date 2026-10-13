from vector_db import VectorDBManager
from litellm_client import LiteLLMClient
from typing import Dict, Any, List

class PolicyRAGAgent:
    """
    Policy RAG Agent responsible for retrieving relevant chunks from ChromaDB
    and generating synthesized policy answers.
    """
    def __init__(self, vector_db: VectorDBManager, llm_client: LiteLLMClient):
        self.vector_db = vector_db
        self.llm = llm_client

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        return self.vector_db.search_policy(query, top_k=top_k)

    def answer_policy_query(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Guard: if no chunks retrieved, return honest message instead of hallucinating
        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find any relevant policy information for your query. "
                    "Please ensure the company policy document has been uploaded, or rephrase your question."
                ),
                "sources": [],
                "chunks_used": 0
            }

        context_text = "\n\n".join([
            f"--- Chunk (Similarity: {c['similarity']}) ---\n{c['text']}"
            for c in retrieved_chunks
        ])

        system_prompt = (
            "You are an expert HR Policy Assistant. Answer the user's question accurately using ONLY the provided policy context.\n"
            "If exact numbers (e.g. leave days, rules) are specified in the context, quote them clearly.\n"
            "If the context does not contain the answer, say so honestly instead of guessing.\n"
            "Be clear, structured, and helpful."
        )

        prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Company Policy Context:\n{context_text}\n\n"
            f"Provide a comprehensive, direct answer based on the context. "
            f"Do not invent information not present in the context."
        )

        response = self.llm.generate_completion(prompt, system_prompt, use_fast_model=False)
        return {
            "answer": response,
            "sources": [c['metadata'].get("document_name", "Policy Document") for c in retrieved_chunks],
            "chunks_used": len(retrieved_chunks)
        }
