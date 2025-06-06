from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import torch
import os
import pickle
import json
from pathlib import Path
from underthesea import word_tokenize
from config import EMBEDDING_MODEL


class HybridIndexer:
    """Unified indexer for both text documents and images using hybrid search."""
    
    def __init__(self, all_sections=None, persist_directory='db', bm25_directory='bm25', 
                 image_metadata_file="data/images/metadata.json"):
        """Initialize the hybrid indexer with text and image support.
        
        Args:
            all_sections: List of Document objects for text indexing.
            persist_directory (str): Directory to persist vector database.
            bm25_directory (str): Directory to persist BM25 indices.
            image_metadata_file (str): Path to image metadata JSON file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        
        # Text BM25 index
        self.bm25_directory = bm25_directory
        self.bm25_path = os.path.join(self.bm25_directory, "bm25_index.pkl")
        self.sections_path = os.path.join(self.bm25_directory, "sections.pkl")
        self.bm25_index = None
        self.doc_id_to_index = {}
        
        # Image indexing
        self._image_metadata_file = Path(image_metadata_file)
        self._image_bm25_path = os.path.join(self.bm25_directory, "image_bm25_index.pkl")
        self._image_data_path = os.path.join(self.bm25_directory, "image_data.pkl")
        self._image_bm25_index = None
        self._images = []
        self._image_tokenized_docs = []

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for Vietnamese and English.
        
        Args:
            text (str): Text to tokenize.
            
        Returns:
            List[str]: List of tokens.
        """
        if not text:
            return []
        
        text = text.lower().strip()
        
        # Try Vietnamese tokenization first
        try:
            tokens = word_tokenize(text)
            # If Vietnamese tokenization produces single characters, fallback to simple split
            if len(tokens) > 1 and all(len(token) == 1 for token in tokens):
                tokens = text.split()
        except:
            # Fallback to simple tokenization
            tokens = text.split()
        
        # Filter out empty tokens and single characters (except meaningful ones)
        meaningful_single_chars = {'a', 'i', 'o', 'u'}
        tokens = [token for token in tokens 
                 if token and (len(token) > 1 or token in meaningful_single_chars)]
        
        return tokens

    def _create_searchable_text(self, image_data: Dict[str, Any]) -> str:
        """Create searchable text from image metadata.
        
        Args:
            image_data (Dict): Image metadata dictionary.
            
        Returns:
            str: Combined searchable text.
        """
        caption = image_data.get('caption', '')
        tags = image_data.get('tags', [])
        
        # Combine caption and tags
        searchable_parts = [caption]
        if tags:
            searchable_parts.extend(tags)
        
        return ' '.join(searchable_parts)

    def _load_image_metadata(self) -> bool:
        """Load image metadata from JSON file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._image_metadata_file.exists():
                print(f"Image metadata file not found: {self._image_metadata_file}")
                return False
            
            with open(self._image_metadata_file, 'r', encoding='utf-8') as f:
                self._images = json.load(f)
            
            print(f"Loaded {len(self._images)} images from metadata")
            return True
        except Exception as e:
            print(f"Error loading image metadata: {e}")
            return False

    def _build_image_index(self) -> bool:
        """Build BM25 index for images.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._images:
                print("No images loaded for indexing.")
                return False
            
            # Create tokenized documents for BM25
            self._image_tokenized_docs = []
            for image in self._images:
                searchable_text = self._create_searchable_text(image)
                tokens = self._tokenize_text(searchable_text)
                self._image_tokenized_docs.append(tokens)
            
            # Build BM25 index
            self._image_bm25_index = BM25Okapi(self._image_tokenized_docs)
            
            print(f"Built image BM25 index for {len(self._image_tokenized_docs)} images")
            return True
        except Exception as e:
            print(f"Error building image BM25 index: {e}")
            return False

    def _save_image_index(self) -> bool:
        """Save image BM25 index to file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._image_bm25_index:
                print("No image index to save.")
                return False
            
            os.makedirs(self.bm25_directory, exist_ok=True)
            
            # Save BM25 index
            with open(self._image_bm25_path, 'wb') as f:
                pickle.dump(self._image_bm25_index, f)
            
            # Save image data and tokenized docs
            image_data = {
                'images': self._images,
                'tokenized_docs': self._image_tokenized_docs
            }
            with open(self._image_data_path, 'wb') as f:
                pickle.dump(image_data, f)
            
            print(f"Saved image index to {self._image_bm25_path}")
            return True
        except Exception as e:
            print(f"Error saving image index: {e}")
            return False

    def _load_image_index(self) -> bool:
        """Load image BM25 index from file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(self._image_bm25_path):
                print(f"Image index file not found: {self._image_bm25_path}")
                return False
            
            # Load BM25 index
            with open(self._image_bm25_path, 'rb') as f:
                self._image_bm25_index = pickle.load(f)
            
            # Load image data
            with open(self._image_data_path, 'rb') as f:
                image_data = pickle.load(f)
            
            self._images = image_data['images']
            self._image_tokenized_docs = image_data['tokenized_docs']
            
            print(f"Loaded image index with {len(self._images)} images")
            return True
        except Exception as e:
            print(f"Error loading image index: {e}")
            return False

    def create(self):
        """Create both text and image indices."""
        # Create directories
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.bm25_directory, exist_ok=True)
        
        # Create text embedding index
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

        # Text BM25: create or load
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

        # Image indexing: create or load
        if not self._load_image_index():
            print("Building new image index...")
            if self._load_image_metadata():
                if self._build_image_index():
                    self._save_image_index()
                    print("✅ Image index created successfully!")
                else:
                    print("❌ Failed to build image index")
            else:
                print("⚠️ No image metadata found, skipping image indexing")

    def hybrid_search(
            self,
            query: str,
            k: int = 10,
            bm25_weight: float = 0.3,
            emb_weight: float = 0.7
        ) -> List[Document]:
        """Perform hybrid search on text documents.
        
        Args:
            query (str): Search query.
            k (int): Number of results to return.
            bm25_weight (float): Weight for BM25 scores.
            emb_weight (float): Weight for embedding scores.
            
        Returns:
            List[Document]: List of relevant documents.
        """
        bm25_docs = self.bm25_index.get_top_n(
            query.split(" "),
            self.all_sections,
            n=k*2
        )
        
        emb_docs = self.vectordb.similarity_search(query, k=k*2)

        # Build rank dictionaries
        # bm25_docs: sorted by score descending (higher score = better, rank 1 is best)
        bm25_ranks = {}
        for rank, doc in enumerate(bm25_docs, 1):
            bm25_ranks[doc.metadata['doc_id']] = rank

        # emb_docs: sorted by score ascending (lower distance = better, rank 1 is best)
        emb_ranks = {}
        for rank, doc in enumerate(emb_docs, 1):
            emb_ranks[doc.metadata['doc_id']] = rank

        # Union of doc_ids from both methods
        all_doc_ids = set(bm25_ranks.keys()).union(set(emb_ranks.keys()))

        doc_map = {doc.metadata['doc_id']: doc for doc in self.all_sections if doc.metadata['doc_id'] in all_doc_ids}
    
        fused_scores = {}
        for doc_id in all_doc_ids:
            rank_bm25 = bm25_ranks.get(doc_id, k*2+1)
            rank_emb = emb_ranks.get(doc_id, k*2+1)
            fused_score = bm25_weight * (1 / rank_bm25) + emb_weight * (1 / rank_emb)
            fused_scores[doc_id] = fused_score

        # Sort and return top k documents
        top_doc_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)[:k]
        return [doc_map[doc_id] for doc_id in top_doc_ids]

    def search_images(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant images using BM25.
        
        Args:
            query (str): Search query.
            k (int): Number of results to return.
            
        Returns:
            List[Dict]: List of relevant image metadata.
        """
        try:
            if not self._image_bm25_index:
                print("Image index not loaded. Call create() first.")
                return []
            
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return []
            
            # Search using BM25
            scores = self._image_bm25_index.get_scores(query_tokens)
            
            # Get top k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    image_data = self._images[idx].copy()
                    image_data['relevance_score'] = float(scores[idx])
                    results.append(image_data)
            
            return results
        except Exception as e:
            print(f"Error searching images: {e}")
            return []

    def search_hybrid_with_images(self, query: str, k_text: int = 5, k_images: int = 3) -> Dict[str, Any]:
        """Perform hybrid search on both text and images.
        
        Args:
            query (str): Search query.
            k_text (int): Number of text results to return.
            k_images (int): Number of image results to return.
            
        Returns:
            Dict[str, Any]: Dictionary containing both text documents and images.
        """
        try:
            # Search text documents
            text_results = self.hybrid_search(query, k=k_text)
            
            # Search images
            image_results = self.search_images(query, k=k_images)
            
            return {
                "text_documents": text_results,
                "images": image_results,
                "total_text_results": len(text_results),
                "total_image_results": len(image_results)
            }
        except Exception as e:
            print(f"Error in hybrid search with images: {e}")
            return {
                "text_documents": [],
                "images": [],
                "total_text_results": 0,
                "total_image_results": 0
            }

    def get_image_stats(self) -> Dict[str, Any]:
        """Get statistics about the image index.
        
        Returns:
            Dict[str, Any]: Statistics about images.
        """
        if not self._images:
            return {"total_images": 0, "index_loaded": False}
        
        # Count tags
        all_tags = []
        for img in self._images:
            all_tags.extend(img.get('tags', []))
        
        unique_tags = set(all_tags)
        
        return {
            "total_images": len(self._images),
            "index_loaded": self._image_bm25_index is not None,
            "unique_tags": len(unique_tags),
            "most_common_tags": list(unique_tags)[:10]  # First 10 tags
        }