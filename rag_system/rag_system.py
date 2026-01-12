"""
RAG System - Complete RAG implementation
Contains all classes for embeddings, vector store, retrieval, and RAG system
"""

import os
import time
import numpy as np
import uuid
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Third-party imports
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================================
# Embedding Manager
# ============================================================================

class Embedding_Manager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding manager
        Args:
            model_name (str): Huggingface model that we have used, it can be some other model as well.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            start_time = time.time()
            print(f"Loading embedding model: {self.model_name} (this may take 10-30 seconds on first run)...")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            self.model = SentenceTransformer(self.model_name)
            elapsed = time.time() - start_time
            print(f"✅ Model loaded successfully in {elapsed:.2f} seconds.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise 

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        Args:
            texts (List[str]): List of texts to be embedded.
        Returns:
            np.ndarray: Array of embeddings.
        """   
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


# ============================================================================
# Vector Store DB
# ============================================================================

class VectorStoreDB:
    """Manages document embedding in a chromaDB vector Store """

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """Initialize the VectorStoreDB
        Args:
            collection_name (str, optional): Name of the collection in chromaDB. 
                                           If None, uses the latest collection (by chunk count).
            persist_directory (str): Directory to persist the chromaDB data.
        """
        if persist_directory is None:
            # Try to auto-detect where documents are stored
            self.persist_directory = self._find_vector_store_path()
        else:
            self.persist_directory = persist_directory
        
        # Initialize client first
        self.client = None
        self.collection = None
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except Exception as e:
            print(f"❌ Error initializing ChromaDB client: {e}")
            raise
        
        # Determine which collection to use
        if collection_name is None:
            # Find the latest collection (one with most chunks, excluding 'pdf_documents' if others exist)
            self.collection_name = self._find_latest_collection()
        else:
            self.collection_name = collection_name
        
        self._initialize_collection()
    
    def _find_vector_store_path(self) -> str:
        """Find the vector store path, checking multiple possible locations."""
        project_root = Path(__file__).parent.parent
        
        # Possible paths where vector store might be
        possible_paths = [
            project_root / "data" / "Vector_Store",  # Standard location
            project_root.parent / "data" / "Vector_Store",  # Relative from notebook
            Path.cwd() / "data" / "Vector_Store",  # Current working directory
            project_root / "Vector_Store",  # Alternative location
        ]
        
        # Check each path for existing vector stores
        for path in possible_paths:
            if path.exists():
                try:
                    test_client = chromadb.PersistentClient(path=str(path))
                    test_collections = test_client.list_collections()
                    
                    # If any collections exist, use this path
                    if test_collections:
                        print(f"✅ Found vector store with documents at: {path}")
                        return str(path)
                except Exception:
                    continue
        
        # If no existing store found, use standard location
        standard_path = project_root / "data" / "Vector_Store"
        print(f"📁 Using vector store location: {standard_path}")
        return str(standard_path)

    def _find_latest_collection(self) -> str:
        """Find the latest collection (collection with most chunks, prioritizing non-default collections).
        
        Returns:
            str: Name of the latest collection
        """
        try:
            collections = self.client.list_collections()
            
            if not collections:
                # No collections exist, use default
                return "pdf_documents"
            
            # Get all collections with their chunk counts
            collection_info = []
            for collection in collections:
                try:
                    count = collection.count()
                    collection_info.append({
                        'name': collection.name,
                        'count': count,
                        'is_default': collection.name == 'pdf_documents'
                    })
                except Exception:
                    continue
            
            if not collection_info:
                return "pdf_documents"
            
            # Sort by: 1) non-default collections first, 2) chunk count (descending)
            collection_info.sort(key=lambda x: (x['is_default'], -x['count']))
            
            # Prefer non-default collections with chunks, otherwise use the one with most chunks
            latest_collection = collection_info[0]['name']
            latest_count = collection_info[0]['count']
            
            print(f"📦 Found {len(collection_info)} collection(s). Using latest: '{latest_collection}' ({latest_count} chunks)")
            
            return latest_collection
            
        except Exception as e:
            print(f"⚠️  Error finding latest collection: {e}. Using default 'pdf_documents'")
            return "pdf_documents"
    
    def _initialize_collection(self):
        """Initialize the ChromaDB collection."""
        try:
            start_time = time.time()
            print(f"Initializing ChromaDB collection: '{self.collection_name}'...")
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG"}
            )
            elapsed = time.time() - start_time
            chunk_count = self.collection.count()
            print(f"✅ ChromaDB collection '{self.collection_name}' initialized ({chunk_count} chunks) in {elapsed:.2f} seconds.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB collection: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the collection.
        Args:
            documents (List[Any]): List of documents to be added (LangChain Document objects).
            embeddings (np.ndarray): Corresponding embeddings for the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
        
        print(f"Adding {len(documents)} chunks to the vector store...")

        ids = []
        metadatas = []
        document_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content) if hasattr(doc, 'page_content') else len(str(doc))
            metadatas.append(metadata)

            document_text.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_text,
                embeddings=embedding_list
            )
            print(f"Successfully added {len(documents)} chunks to the vector store.")
        except Exception as e:
            print(f"Error adding chunks to vector store: {e}")
            raise
    
    def delete_chunks_by_document(self, document_name: str = None, document_filename: str = None):
        """Delete all chunks belonging to a specific document.
        
        Args:
            document_name (str, optional): Document name (without extension) to delete
            document_filename (str, optional): Document filename to delete
        
        Returns:
            int: Number of chunks deleted
        """
        if not document_name and not document_filename:
            raise ValueError("Either document_name or document_filename must be provided")
        
        try:
            # Build where clause to find chunks from this document
            where_clause = {
                "$or": []
            }
            
            if document_name:
                where_clause["$or"].append({"document_name": document_name})
            
            if document_filename:
                where_clause["$or"].append({"document_filename": document_filename})
            
            # Get all chunks matching this document
            results = self.collection.get(
                where=where_clause,
                include=["metadatas"]
            )
            
            if results and results['ids']:
                chunk_ids = results['ids']
                self.collection.delete(ids=chunk_ids)
                deleted_count = len(chunk_ids)
                doc_identifier = document_name or document_filename
                print(f"✅ Deleted {deleted_count} chunks from document: {doc_identifier}")
                return deleted_count
            else:
                doc_identifier = document_name or document_filename
                print(f"ℹ️  No chunks found for document: {doc_identifier}")
                return 0
                
        except Exception as e:
            print(f"❌ Error deleting chunks by document: {e}")
            raise


# ============================================================================
# RAG Retrieval
# ============================================================================

class RAGRetrieval:
    """Handles query based retrieval from the vector store."""

    def __init__(self, vector_store, embedding_manager):
        """Initialize the retriever
        Args:  
            vector_store: Vector store containing the document embeddings
            embedding_manager: Embedding manager to generate query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int=5, score_threshold: float = 0.0, document_name: str = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a given query.
        Args:
            query (str): The input query string.
            top_k (int): Number of top chunks to retrieve.
            score_threshold (float): Minimum similarity score to consider.
            document_name (str, optional): Filter results to a specific document name/filename. 
                                          If None, retrieves from all documents.
            
        Returns:
            List[Dict[str, Any]]: List of retrieved chunks with metadata and scores."""
        
        print(f"Retrieval of chunks for query: '{query}'" + (f" (filtered by document: {document_name})" if document_name else ""))

        # Check if vector store has documents
        doc_count = self.vector_store.collection.count()
        if doc_count == 0:
            print("⚠️  WARNING: Vector store is empty! No chunks found.")
            return []

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # Build where clause for document filtering if specified
            where_clause = None
            if document_name:
                where_clause = {
                    "$or": [
                        {"document_name": document_name},
                        {"document_filename": document_name}
                    ]
                }
            
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, doc_count),  # Don't request more than available
                where=where_clause,  # Filter by document if specified
                include=["embeddings", "documents", "metadatas", "distances"]
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                ids = results.get('ids', [[]])[0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })
                
                if retrieved_docs:
                    # Group by document for summary
                    doc_summary = {}
                    for doc in retrieved_docs:
                        doc_name = doc.get('metadata', {}).get('document_name', 'Unknown')
                        doc_summary[doc_name] = doc_summary.get(doc_name, 0) + 1
                    
                    summary = ", ".join([f"{name}: {count}" for name, count in doc_summary.items()])
                    print(f"Retrieved {len(retrieved_docs)} chunks from {len(doc_summary)} document(s) (top score: {retrieved_docs[0]['score']:.4f}).")
                    print(f"  Document breakdown: {summary}")
                else:
                    print(f"No chunks passed score threshold ({score_threshold}).")
                    print(f"Available chunks had scores below threshold.")
            else:
                print("No chunks retrieved from vector store.")

            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            raise


# ============================================================================
# RAG System
# ============================================================================

class RAGSystem:
    """Complete RAG system combining retrieval and generation."""
    
    # Default structured prompt template for consistent answer formatting
    DEFAULT_STRUCTURED_PROMPT = """You are a professor. Based on the provided context from various documents, answer the user's question following this structured format:

1. Start with a clear, concise answer to the question and summarise it.

2. Examples: Provide real-life examples to support your answer.

3. Sources: Mention which document(s) the information comes from when relevant.

IMPORTANT: The context is organized by document. Each section is labeled with its source document name. Use this information to provide accurate, document-specific answers and cite sources when relevant.

Context (organized by document):
{context}

Question:
{query}

Answer (follow the structured format above):
"""
    
    def __init__(self, use_structured_prompt: bool = True):
        """Initialize the RAG system with all components.
        
        Args:
            use_structured_prompt (bool): If True, uses structured prompt template for consistent formatting (default: True)
        """
        total_start = time.time()
        print("=" * 60)
        print("Initializing RAG System...")
        print("=" * 60)
        
        print("\n📦 Step 1/3: Loading embedding model...")
        start = time.time()
        self.embedding_manager = Embedding_Manager()
        print(f"   ✓ Completed in {time.time() - start:.2f}s\n")
        
        print("📦 Step 2/3: Initializing vector store...")
        start = time.time()
        self.vector_store = VectorStoreDB()
        print(f"   ✓ Completed in {time.time() - start:.2f}s\n")
        
        print("📦 Step 3/3: Setting up retrieval and LLM...")
        start = time.time()
        self.retriever = RAGRetrieval(self.vector_store, self.embedding_manager)
        
        # Initialize LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")
        # Strip whitespace in case of formatting issues
        google_api_key = google_api_key.strip()
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        
        # Set default prompt template
        self.use_structured_prompt = use_structured_prompt
        self.default_prompt_template = self.DEFAULT_STRUCTURED_PROMPT if use_structured_prompt else self._get_simple_prompt_template()
        
        print(f"   ✓ Completed in {time.time() - start:.2f}s\n")
        
        total_time = time.time() - total_start
        print("=" * 60)
        print(f"✅ RAG System initialized successfully in {total_time:.2f} seconds!")
        print("=" * 60)
    
    def _get_simple_prompt_template(self) -> str:
        """Get simple prompt template (backward compatible)."""
        return """Use the following context to answer the question.
        Context: {context}
        Question: {query}
    """
    
    def set_prompt_template(self, prompt_template: str):
        """Set a custom default prompt template for all queries.
        
        Args:
            prompt_template (str): Prompt template with {context} and {query} placeholders
        """
        # Validate template has required placeholders
        try:
            test_format = prompt_template.format(context="test", query="test")
            self.default_prompt_template = prompt_template
            print("✅ Default prompt template updated successfully.")
        except KeyError as e:
            raise ValueError(
                f"Prompt template must contain {{context}} and {{query}} placeholders. "
                f"Missing placeholder: {e}"
            )
    
    def query(self, query: str, top_k: int = 3) -> str:
        """Query the RAG system using the default prompt template.
        
        Args:
            query (str): User query
            top_k (int): Number of chunks to retrieve (default: 3)
        
        Returns:
            str: Generated answer with consistent formatting
        """
        # Check vector store first
        doc_count = self.vector_store.collection.count()
        if doc_count == 0:
            return "I couldn't find any documents in the vector store. Please make sure documents have been loaded by running the notebook cells first."
        
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, top_k=top_k, score_threshold=0.0)
        
        if not results:
            return "I couldn't find any relevant information in the documents to answer your question. Try rephrasing your query or asking about a different topic."
        
        # Group chunks by document for better organization
        context = self._format_context_by_document(results)
        
        # Format the prompt template with context and query
        try:
            formatted_prompt = self.default_prompt_template.format(context=context, query=query)
        except KeyError as e:
            raise ValueError(
                f"Prompt template must contain {{context}} and {{query}} placeholders. "
                f"Missing placeholder: {e}"
            )
        
        # Generate answer using LLM
        response = self.llm.invoke(formatted_prompt)
        return response.content
    
    def _format_context_by_document(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks grouped by document for better context organization.
        
        Args:
            results: List of retrieved chunk dictionaries with metadata
        
        Returns:
            str: Formatted context string grouped by document
        """
        # Group chunks by document
        documents_dict = {}
        for chunk in results:
            # Get document name from metadata
            doc_name = chunk.get('metadata', {}).get('document_name', 'Unknown Document')
            doc_filename = chunk.get('metadata', {}).get('document_filename', 'unknown.pdf')
            
            # Use document name as key, or filename if name not available
            doc_key = doc_name if doc_name != 'Unknown Document' else doc_filename
            
            if doc_key not in documents_dict:
                documents_dict[doc_key] = {
                    'filename': doc_filename,
                    'chunks': []
                }
            
            # Add chunk with its metadata
            documents_dict[doc_key]['chunks'].append({
                'content': chunk['document'],
                'chunk_index': chunk.get('metadata', {}).get('chunk_index', 0),
                'score': chunk.get('score', 0.0),
                'rank': chunk.get('rank', 0)
            })
        
        # Format context grouped by document
        context_parts = []
        for doc_key, doc_info in documents_dict.items():
            # Document header
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"📄 Document: {doc_key} ({doc_info['filename']})")
            context_parts.append(f"{'='*60}\n")
            
            # Sort chunks by rank (relevance order)
            sorted_chunks = sorted(doc_info['chunks'], key=lambda x: x['rank'])
            
            # Add chunks from this document
            for chunk_info in sorted_chunks:
                context_parts.append(f"Chunk {chunk_info['chunk_index']} (Relevance: {chunk_info['score']:.3f}):")
                context_parts.append(chunk_info['content'])
                context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Embedding_Manager",
    "VectorStoreDB",
    "RAGRetrieval",
    "RAGSystem"
]
