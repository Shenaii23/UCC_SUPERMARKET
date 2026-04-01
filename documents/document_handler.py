"""
Document Handler for Terms and Conditions, Privacy Policy, and LLM Guidelines
Handles loading, caching, and retrieving relevant sections from policy PDFs
"""
import pdfplumber
import hashlib
import json
import os
from typing import Dict, List, Optional

# Document paths
DOCUMENTS = {
    "terms_and_conditions": "documents/UCC_Terms_Full.pdf",
    "privacy_policy": "documents/UCC_Privacy_Full.pdf",
    "llm_guidelines": "documents/UCC_LLM_Full.pdf",
    "store_info": "documents/UCC_Store_Info.pdf",
    "general_chat": "documents/General_Chat_Guide.pdf"
}

# Cache file to store document content and hashes
CACHE_FILE = "documents/.document_cache.json"


class DocumentManager:
    """Manages loading, caching, and retrieving policy documents"""
    
    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.document_hashes: Dict[str, str] = {}
        self.load_documents()
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")
            return ""
    
    def load_cache(self) -> Dict:
        """Load cached documents and hashes"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def save_cache(self):
        """Save documents and hashes to cache"""
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            cache_data = {
                "hashes": self.document_hashes,
                "documents": self.documents
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            print("[CACHE] Documents cached successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_pdf_text(self, file_path: str) -> str:
        """Load text from a PDF file"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return ""
    
    def load_documents(self):
        """Load documents with caching and change detection"""
        cache = self.load_cache()
        cached_hashes = cache.get("hashes", {})
        
        print("[DOCUMENT] Loading policy documents...")
        
        for doc_name, doc_path in DOCUMENTS.items():
            if not os.path.exists(doc_path):
                print(f"[WARNING] Document not found: {doc_path}")
                continue
            
            # Check if file has changed
            current_hash = self.get_file_hash(doc_path)
            
            if doc_name in cached_hashes and cached_hashes[doc_name] == current_hash:
                # Use cached version
                self.documents[doc_name] = cache.get("documents", {}).get(doc_name, "")
                print(f"[CACHE] Using cached version: {doc_name}")
            else:
                # Load fresh from PDF
                print(f"[DOCUMENT] Loading fresh: {doc_name}")
                text = self.load_pdf_text(doc_path)
                if text:
                    self.documents[doc_name] = text
                    self.document_hashes[doc_name] = current_hash
                    print(f"[DOCUMENT] Loaded {len(text)} characters from {doc_name}")
        
        # Save updated cache
        if self.documents:
            self.save_cache()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        Overlaps help maintain context between chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def search_document(self, doc_name: str, query: str, top_k: int = 3) -> str:
        """
        Search a document for relevant sections based on keyword matching.
        Returns the most relevant chunks.
        """
        if doc_name not in self.documents:
            return f"Document '{doc_name}' not found."
        
        document_text = self.documents[doc_name]
        chunks = self.chunk_text(document_text)
        
        # Normalize query
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score chunks by keyword matches
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count how many query words appear in this chunk
            matches = sum(1 for word in query_words if word in chunk_lower)
            if matches > 0:
                scored_chunks.append((chunk, matches))
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = scored_chunks[:top_k]
        
        if not relevant_chunks:
            # If no keyword matches, return first chunk as fallback
            return chunks[0] if chunks else "No content found."
        
        # Combine relevant chunks
        result = "\n---\n".join([chunk for chunk, _ in relevant_chunks])
        return result
    
    def get_answer_context(self, user_query: str) -> str:
        """
        Determine which documents are relevant and retrieve appropriate sections.
        Returns formatted context for the LLM.
        """
        query_lower = user_query.lower()
        context_parts = []
        
        # Map keywords to documents
        keyword_mapping = {
            "terms_and_conditions": ["terms", "conditions", "policy", "service", "agreement"],
            "privacy_policy": ["privacy", "data", "personal", "information", "collect"],
            "llm_guidelines": ["ai", "llm", "algorithm", "automated", "decision"],
            "store_info": ["store", "hours", "location", "address", "phone", "departments"],
            "general_chat": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "good night"]
        }
        
        # Find relevant documents
        relevant_docs = []
        for doc_name, keywords in keyword_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_docs.append(doc_name)
        
        # If no specific match, search all documents
        if not relevant_docs:
            relevant_docs = list(DOCUMENTS.keys())
        
        # Retrieve relevant sections from each document
        for doc_name in relevant_docs:
            section = self.search_document(doc_name, user_query, top_k=2)
            if section and section != "Document not found.":
                # Format the section
                context_parts.append(f"### {doc_name.replace('_', ' ').title()}\n{section}")
        
        if not context_parts:
            return "No policy documents available."
        
        return "\n\n".join(context_parts)


# Initialize global document manager
document_manager = DocumentManager()


def get_document_context(user_query: str) -> str:
    """
    Public function to get relevant document context for a user query.
    Returns formatted policy text for the LLM.
    """
    return document_manager.get_answer_context(user_query)