from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
import uuid
import hashlib
from typing import List
from config import GLOB_PATTERN, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentChunker:
    def __init__(self, dir_path, glob_pattern=GLOB_PATTERN, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.dir_path = dir_path
        self.glob_pattern = glob_pattern
        self.documents = []
        self.all_sections = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        """Load markdown documents from the specified directory."""
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(
            self.dir_path,
            glob=self.glob_pattern,
            show_progress=True,
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs
        )
        self.documents = loader.load()

    def _generate_chunk_id(self, text, metadata):
        # Use a hash of the text and metadata for a stable, unique ID
        base = text + str(metadata)
        return hashlib.md5(base.encode('utf-8')).hexdigest()
    
    def smart_chunk_markdown(self, markdown: str) -> List[dict]:
        """Hierarchically splits markdown by #, ##, ### headers, then uses LangChain splitter to ensure all chunks < max_len.
        Returns a list of dictionaries with text and metadata."""
        
        # Initialize splitters
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3")
            ],
            strip_headers=False  # Keep headers in content for context
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []

        # Split by headers
        chunks = []
        header_docs = header_splitter.split_text(markdown)
        has_headers = bool(header_docs and any(doc.metadata for doc in header_docs))

        if has_headers:
            for doc in header_docs:
                if len(doc.page_content) > self.chunk_size:
                    split_texts = char_splitter.split_text(doc.page_content)
                    for text in split_texts:
                        if text.strip():
                            meta = doc.metadata.copy()
                            chunk_id = self._generate_chunk_id(text, meta)
                            chunks.append({
                                "text": text,
                                "metadata": meta,
                                "chunk_id": chunk_id
                            })
                elif doc.page_content.strip():
                    meta = doc.metadata.copy()
                    chunk_id = self._generate_chunk_id(doc.page_content, meta)
                    chunks.append({
                        "text": doc.page_content,
                        "metadata": meta,
                        "chunk_id": chunk_id
                    })
        else:
            meta = {}
            split_texts = char_splitter.split_text(markdown)
            for text in split_texts:
                if text.strip():
                    chunk_id = self._generate_chunk_id(text, meta)
                    chunks.append({
                        "text": text,
                        "metadata": meta,
                        "chunk_id": chunk_id
                    })

        if not chunks and markdown.strip():
            meta = {}
            chunk_id = self._generate_chunk_id(markdown.strip()[:self.chunk_size], meta)
            chunks.append({
                "text": markdown.strip()[:self.chunk_size],
                "metadata": meta,
                "chunk_id": chunk_id
            })

        return [
            {"text": chunk["text"], "metadata": chunk["metadata"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks if chunk["text"].strip()
        ]
    
    
    def split_documents(self):
        """Split markdown documents into chunks using smart_chunk_markdown, storing in all_sections."""
        for doc in self.documents:


            # Split markdown using smart_chunk_markdown
            chunks = self.smart_chunk_markdown(
                markdown=doc.page_content,
                max_len=self.max_len
            )

            # Convert chunks to LangChain Document objects and add to all_sections
            for chunk in chunks:
                section = Document(
                    page_content=chunk["text"],
                    metadata={
                        'doc_id': str(uuid.uuid4()),
                    }
                )
                self.all_sections.append(section)