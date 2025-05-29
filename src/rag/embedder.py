from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import torch
import os
import pickle
from config import EMBEDDING_MODEL

class HybridIndexer:
    def __init__(self, all_sections=None, persist_directory='db', bm25_directory='bm25'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        self.bm25_directory = bm25_directory
        self.bm25_path = os.path.join(self.bm25_directory, "bm25_index.pkl")
        self.sections_path = os.path.join(self.bm25_directory, "sections.pkl")
        self.bm25_index = None
        self.doc_id_to_index = {}

    def create(self):
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": self.device}
        )
        # Chroma: create or load
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding
            )
        elif self.all_sections:
            self.vectordb = Chroma.from_documents(
                documents=self.all_sections,
                embedding=embedding,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError("No documents provided to create a new index.")

        # BM25: create or load
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)
        elif self.all_sections:
            texts = [doc.page_content.split(" ") for doc in self.all_sections]
            self.bm25_index = BM25Okapi(texts)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25_index, f)
        else:
            raise ValueError("No documents provided to create a new BM25 index.")

        # Persist all_sections for later retrieval if available
        if self.all_sections:
            with open(self.sections_path, "wb") as f:
                pickle.dump(self.all_sections, f)
            for idx, doc in enumerate(self.all_sections):
                self.doc_id_to_index[doc.metadata['doc_id']] = idx
        elif os.path.exists(self.sections_path):
            with open(self.sections_path, "rb") as f:
                self.all_sections = pickle.load(f)
            for idx, doc in enumerate(self.all_sections):
                self.doc_id_to_index[doc.metadata['doc_id']] = idx
        else:
            self.all_sections = None  # Explicitly set to None

    def bm25_search(self, query: str, k: int=10) -> List[Tuple[Document, float]]:
        """Find top k similarity document using BM25"""
        query_tokens = query.split(" ")
        scores = self.bm25_index.get_scores(query_tokens)
        min_score = min(scores)
        max_score = max(scores)
        # Normalize scores to [0, 1]
        
        if max_score - min_score > 0:
            norm_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            norm_scores = [1.0 for _ in scores]
        top_k_idx = sorted(range(len(norm_scores)), key=lambda i: norm_scores[i], reverse=True)[:k]
        return [(self.all_sections[i], norm_scores[i]) for i in top_k_idx]

    def embedding_search(self, query:str, k:int=10) -> List[Tuple[Document, float]]:
        """Find top k similarity document using chroma vector search"""
        return self.vectordb.similarity_search_with_score(query, k=k)

    def hybrid_search(
            self,
            query: str,
            k: int=10,
            bm25_weight: float=0.3,
            emb_weight: float=0.7,
            score_threshold: float=0.0
        ) -> List[Document]:
        bm25_docs = self.bm25_search(query, k=k*2)
        emb_docs = self.embedding_search(query, k=k*2)

        # Build rank dictionaries
        bm25_ranks = {doc.metadata['doc_id']: rank+1 for rank, (doc, _) in enumerate(bm25_docs)}
        emb_ranks = {doc.metadata['doc_id']: rank+1 for rank, (doc, _) in enumerate(emb_docs)}
        
        # Union of doc_ids from both methods
        all_doc_ids = set(bm25_ranks.keys()) | set(emb_ranks.keys())

        doc_map = {doc.metadata['doc_id']: doc for doc, _ in bm25_docs + emb_docs}

        fused_scores = {}
        for doc_id in all_doc_ids:
            rank_bm25 = bm25_ranks.get(doc_id, k*2+1)
            rank_emb = emb_ranks.get(doc_id, k*2+1)
            fused_score = bm25_weight * (1 / rank_bm25) + emb_weight * (1 / rank_emb)
            if fused_score >= score_threshold:
                fused_scores[doc_id] = fused_score

        # Sort and return top k documents passing the threshold
        top_doc_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)[:k]
        return [doc_map[doc_id] for doc_id in top_doc_ids]
