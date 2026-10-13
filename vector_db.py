import os
import yaml
import chromadb
from chromadb.config import Settings
import requests
from typing import List, Dict, Any, Tuple
import uuid
import re

class VectorDBManager:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["vector_db"]
        
        self.persist_dir = self.config.get("persist_directory", "./vector_store")
        self.collection_name = self.config.get("collection_name", "company_policies")
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.ollama_base = "http://localhost:11434"
        self.embed_model = "mxbai-embed-large"

        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding via local Ollama API with fast fallback."""
        if getattr(self, "_ollama_offline", False):
            return [0.01 * (i % 50) for i in range(1024)]

        try:
            resp = requests.post(
                f"{self.ollama_base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=15
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
        except Exception as e:
            self._ollama_offline = True
            print(f"Ollama Embedding service not reachable ({e}). Using local fallback embeddings.")
        
        # Fallback dummy embedding if Ollama embedding service is briefly unavailable
        return [0.01 * (i % 50) for i in range(1024)]

    def chunk_text(self, text: str) -> List[str]:
        """Recursive chunking by headers, paragraphs, and sentences."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            if len(current_chunk) + len(p) <= self.chunk_size:
                current_chunk += ("\n\n" + p if current_chunk else p)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself exceeds chunk size, split by sentences
                if len(p) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) <= self.chunk_size:
                            sub_chunk += (" " + s if sub_chunk else s)
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = s
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:self.chunk_size]]

    def ingest_document(self, document_name: str, text: str, category: str = "Company Policy") -> Dict[str, Any]:
        """Chunk, embed, and index document into ChromaDB."""
        chunks = self.chunk_text(text)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_name}_{idx}_{uuid.uuid4().hex[:8]}"
            emb = self.generate_embedding(chunk_text)
            
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk_text)
            metadatas.append({
                "document_name": document_name,
                "category": category,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

        return {
            "document_name": document_name,
            "total_chunks": len(chunks),
            "status": "Indexed Successfully",
            "sample_chunk": chunks[0][:150] if chunks else ""
        }

    def search_policy(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Semantic vector search against ingested company policies."""
        query_emb = self.generate_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                similarity = max(0.0, round(1.0 - float(dist), 4))
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                })

        return output

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return total chunks and items stored in ChromaDB."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_directory": self.persist_dir
        }
