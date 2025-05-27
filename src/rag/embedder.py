from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import torch
import os
import pickle
from config import EMBEDDING_MODEL


class HybridIndexer:
    def __init__(self, all_sections=None, persist_directory='db', tfidf_directory='tfidf'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        self.tfidf_directory = tfidf_directory
        self.tfidf_path = os.path.join(self.tfidf_directory, "tfidf.pkl")
        self.sections_path = os.path.join(self.tfidf_directory, "sections.pkl")
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
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

        # TF-IDF: create or load
        if os.path.exists(self.tfidf_path):
            with open(self.tfidf_path, "rb") as f:
                self.tfidf_vectorizer, self.tfidf_matrix = pickle.load(f)
        elif self.all_sections:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=None,
                lowercase=True,
                norm='l2',
                sublinear_tf=True
            )
            texts = [doc.page_content for doc in self.all_sections]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            with open(self.tfidf_path, "wb") as f:
                pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix), f)
        else:
            raise ValueError("No documents provided to create a new TF-IDF index.")

        # Persist all_sections for later retrieval if available
       
        if self.all_sections:
            with open(self.sections_path, "wb") as f:
                pickle.dump(self.all_sections, f)
            for idx, doc in enumerate(self.all_sections):
                self.doc_id_to_index[doc.metadata['doc_id']] = idx
        # If not available, try to load from disk
        elif os.path.exists(self.sections_path):
            with open(self.sections_path, "rb") as f:
                self.all_sections = pickle.load(f)
            for idx, doc in enumerate(self.all_sections):
                self.doc_id_to_index[doc.metadata['doc_id']] = idx
        else:
            self.all_sections = None  # Explicitly set to None


    def bm25_search(self, query: str, k: int=10) -> List[Tuple[Document, float]]:
        """Find top k similarity document using BM25"""
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_k_idx = scores.argsort()[::-1][:k]
        return [(self.all_sections[i], scores[i]) for i in top_k_idx]

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
        # Combine and deduplicate using reciprocal rank fusion
        doc_scores = {}
        for doc, score in bm25_docs:
            doc_id = doc.metadata['doc_id']
            score = score * bm25_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_scores[(doc_id, 'doc')] = doc
            
        for doc, score in emb_docs:
            doc_id = doc.metadata['doc_id']
            score = score * emb_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_scores[(doc_id, 'doc')] = doc
            
        # Filter by score threshold
        filtered_docs = [
            doc_scores[(doc_id, 'doc')]
            for doc_id in doc_scores
            if isinstance(doc_id, str) and doc_scores[doc_id] >= score_threshold
        ]
           
        # Sort by combined score and return top k
        filtered_docs.sort(
            key=lambda doc: doc_scores[doc.metadata['doc_id']],
            reverse=True
        )
        
        return filtered_docs[:k]