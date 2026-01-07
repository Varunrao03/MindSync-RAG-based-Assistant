"""
RAG System - Complete RAG implementation based on notebook
Contains all classes and functions from Detailed_RAG_implementation.ipynb
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
from langchain_groq import ChatGroq

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

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = None):
        """Initialize the VectorStoreDB
        Args:
            collection_name (str): Name of the collection in chromaDB.
            persist_directory (str): Directory to persist the chromaDB data.
        """
        self.collection_name = collection_name
        if persist_directory is None:
            # Try to auto-detect where documents are stored
            self.persist_directory = self._find_vector_store_path()
        else:
            self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _find_vector_store_path(self) -> str:
        """Find the vector store path, checking multiple possible locations."""
        project_root = Path(__file__).parent.parent
        
        # Possible paths where vector store might be
        possible_paths = [
            project_root / "data" / "Vector_Store",  # Standard location (from our code)
            project_root.parent / "data" / "Vector_Store",  # Relative from notebook (../data/Vector_Store)
            Path.cwd() / "data" / "Vector_Store",  # Current working directory
            project_root / "Vector_Store",  # Alternative location
        ]
        
        # Check each path for existing collections with documents
        for path in possible_paths:
            if path.exists():
                try:
                    test_client = chromadb.PersistentClient(path=str(path))
                    test_collections = test_client.list_collections()
                    
                    # Check if our collection exists and has documents
                    for collection in test_collections:
                        if collection.name == self.collection_name:
                            if collection.count() > 0:
                                print(f"✅ Found vector store with documents at: {path}")
                                return str(path)
                except Exception:
                    continue
        
        # If no existing store found, use standard location
        standard_path = project_root / "data" / "Vector_Store"
        print(f"📁 Using vector store location: {standard_path}")
        return str(standard_path)

    def _initialize_store(self):
        """Initialize the chromaDB client and collection."""
        try:
            start_time = time.time()
            print("Initializing ChromaDB client...")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG"}
            )
            elapsed = time.time() - start_time
            print(f"✅ ChromaDB client and collection '{self.collection_name}' initialized in {elapsed:.2f} seconds.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the collection.
        Args:
            documents (List[Any]): List of documents to be added (LangChain Document objects).
            embeddings (np.ndarray): Corresponding embeddings for the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")
        
        print(f"Adding {len(documents)} documents to the vector store...")

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
            print(f"Successfully added {len(documents)} documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
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

    def retrieve(self, query: str, top_k: int=5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a given query.
        Args:
            query (str): The input query string.
            top_k (int): Number of top documents to retrieve.
            score_threshold (float): Minimum similarity score to consider.
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with metadata and scores."""
        
        print(f"Retrieval of documents for query: '{query}'")

        # Check if vector store has documents
        doc_count = self.vector_store.collection.count()
        if doc_count == 0:
            print("⚠️  WARNING: Vector store is empty! No documents found.")
            return []

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, doc_count),  # Don't request more than available
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
                    print(f"Retrieved {len(retrieved_docs)} documents (top score: {retrieved_docs[0]['score']:.4f}).")
                else:
                    print(f"No documents passed score threshold ({score_threshold}).")
                    print(f"Available documents had scores below threshold.")
            else:
                print("No documents retrieved from vector store.")

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
    DEFAULT_STRUCTURED_PROMPT = """You are a professor. Based on the provided context, answer the user's question following this structured format:

1. Start with a clear, concise answer to the question and summarise it.

2. Examples: Provide real-life examples to support your answer.

Context:
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
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        
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
            top_k (int): Number of documents to retrieve (default: 3)
        
        Returns:
            str: Generated answer with consistent formatting
        """
        # Check vector store first
        doc_count = self.vector_store.collection.count()
        if doc_count == 0:
            return "I couldn't find any documents in the vector store. Please make sure documents have been loaded by running the notebook cells first."
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(query, top_k=top_k, score_threshold=0.0)
        context = "\n\n".join([doc['document'] for doc in results]) if results else ""
        
        if not context:
            return "I couldn't find any relevant information in the documents to answer your question. Try rephrasing your query or asking about a different topic."
        
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


# ============================================================================
# Singleton Functions
# ============================================================================

rag_system = None

def get_rag_system():
    """Get or initialize the RAG system singleton."""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system


__all__ = [
    "Embedding_Manager",
    "VectorStoreDB",
    "RAGRetrieval",
    "RAGSystem",
    "get_rag_system"
]

