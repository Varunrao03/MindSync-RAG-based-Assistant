"""
FastAPI Backend for RAG System Chatbot
"""

import os
import sys
import shutil
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Add project root to Python path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Load environment variables from .env file
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

from rag_system import RAGSystem, VectorStoreDB, Embedding_Manager, RAGRetrieval
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from quiz_generator import QuizGenerator


# Initialize FastAPI app
app = FastAPI(title="RAG System Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
FRONTEND_DIR = BASE_DIR / "frontend"
PDF_DIR = BASE_DIR / "data" / "pdf"
DATA_DIR = BASE_DIR / "data"
EXTRACTED_TEXTS_DIR = BASE_DIR / "Extracted_texts"
PUBLIC_DIR = BASE_DIR / "public"
PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize RAG system (lazy loading)
rag_system = None
embedding_manager = None
vector_store = None
quiz_generator = None

def get_quiz_generator():
    """Get or initialize quiz generator."""
    global quiz_generator
    if quiz_generator is None:
        quiz_generator = QuizGenerator()
    return quiz_generator

def get_rag():
    """Get or initialize RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system

def reset_rag_system():
    """Reset the RAG system to force reload (useful after adding new documents)."""
    global rag_system
    rag_system = None
    print("🔄 RAG system reset - will reinitialize on next query")

def get_embedding_manager():
    """Get or initialize embedding manager."""
    global embedding_manager
    if embedding_manager is None:
        embedding_manager = Embedding_Manager()
    return embedding_manager

def get_vector_store():
    """Get or initialize vector store."""
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreDB()
    return vector_store

def split_documents(documents, max_characters=1000, overlap=50):
    """Split PDF documents into chunks using unstructured.ai chunking by title.
    
    Args:
        documents: List of LangChain Document objects (PDFs only)
        max_characters: Maximum characters per chunk (default: 1000)
        overlap: Character overlap between chunks (default: 50)
    
    Returns:
        List of LangChain Document objects with chunked content
    """
    chunked_docs = []
    
    for doc in documents:
        # Get the source file path
        source_file = doc.metadata.get('source', '')
        
        # Only process PDF files
        if not source_file or not source_file.endswith('.pdf'):
            print(f"⚠️ Warning: Skipping non-PDF file: {source_file}")
            continue
        
        try:
            # Partition PDF using unstructured
            # Using "fast" strategy for better performance (use "hi_res" for scanned PDFs)
            elements = partition_pdf(
                filename=source_file,
                strategy="fast",  # Fast strategy for better performance
                infer_table_structure=True,
                languages=["eng"]  # Specify English language to suppress warning
            )
            
            # Chunk by title using unstructured
            chunks = chunk_by_title(
                elements=elements,
                max_characters=max_characters,
                overlap=max(overlap, 50)  # Ensure minimum overlap
            )
            
            # Extract document name from source file path
            source_path = Path(source_file)
            document_filename = source_path.name  # e.g., "Attention.pdf"
            document_name = source_path.stem  # e.g., "Attention" (without extension)
            
            # Convert unstructured chunks to LangChain Documents
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=str(chunk),
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'chunk_method': 'unstructured_title_chunking',
                        # Document-wise organization metadata
                        'document_name': document_name,
                        'document_filename': document_filename,
                        'document_source': str(source_file),
                        'total_chunks_in_document': len(chunks)  # Total chunks for this document
                    }
                )
                chunked_docs.append(chunked_doc)
                
        except Exception as e:
            print(f"❌ Error processing PDF {source_file}: {e}")
            raise
    
    print(f"✅ Split {len(documents)} PDF documents into {len(chunked_docs)} chunks using unstructured.ai")
    return chunked_docs

def sanitize_collection_name(filename: str) -> str:
    """Sanitize a filename to create a valid ChromaDB collection name.
    
    Args:
        filename: PDF filename (e.g., "My Document.pdf")
    
    Returns:
        Sanitized collection name (e.g., "my_document")
    """
    # Remove extension
    name = Path(filename).stem
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Convert to lowercase
    name = name.lower()
    # Ensure it's not empty
    if not name:
        name = "document"
    return name

def process_pdf_to_separate_collection(pdf_path: Path) -> Dict:
    """Process a single PDF file into its own separate collection.
    
    Args:
        pdf_path: Path to the PDF file to process.
    
    Returns:
        Dictionary with processing results including collection name.
    """
    if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
        return {
            "success": False,
            "message": f"Invalid PDF file: {pdf_path}",
            "documents_loaded": 0,
            "chunks_created": 0,
            "collection_name": None
        }
    
    # Create collection name from PDF filename
    collection_name = sanitize_collection_name(pdf_path.name)
    print(f"📦 Creating separate collection: '{collection_name}' for PDF: {pdf_path.name}")
    
    # Create a new vector store instance with the specific collection name
    embedding_mgr = get_embedding_manager()
    
    # Use the same persist directory but different collection name
    persist_dir = DATA_DIR / "Vector_Store"
    vs = VectorStoreDB(collection_name=collection_name, persist_directory=str(persist_dir))
    
    # Load the PDF
    loader = PyMuPDFLoader(str(pdf_path))
    pdf_documents = loader.load()
    
    if not pdf_documents:
        return {
            "success": False,
            "message": f"No documents loaded from {pdf_path.name}",
            "documents_loaded": 0,
            "chunks_created": 0,
            "collection_name": collection_name
        }
    
    # Chunk documents
    chunks = split_documents(pdf_documents, max_characters=1000, overlap=50)
    
    # Generate embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_mgr.generate_embeddings(chunk_texts)
    
    # Add to the specific collection
    vs.add_documents(chunks, embeddings)
    
    total_count = vs.collection.count()
    
    print(f"✅ Processed {pdf_path.name} into collection '{collection_name}': {len(chunks)} chunks")
    
    return {
        "success": True,
        "message": f"PDF processed into separate collection successfully",
        "documents_loaded": len(pdf_documents),
        "chunks_created": len(chunks),
        "collection_name": collection_name,
        "total_documents": total_count
    }

def process_and_add_documents(pdf_files: List[Path]) -> Dict:
    """Process PDF files and add them to the vector store.
    
    Args:
        pdf_files: List of specific PDF file paths to process. If empty, processes all PDFs.
    """
    embedding_mgr = get_embedding_manager()
    vs = get_vector_store()
    
    # Load specific PDFs or all PDFs if none specified
    if pdf_files:
        # Load only the specified PDF files
        pdf_documents = []
        for pdf_path in pdf_files:
            if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                loader = PyMuPDFLoader(str(pdf_path))
                docs = loader.load()
                pdf_documents.extend(docs)
    else:
        # Load all PDFs from directory
        dir_loader = DirectoryLoader(
            str(PDF_DIR),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False
        )
        pdf_documents = dir_loader.load()
    
    if not pdf_documents:
        return {
            "success": False,
            "message": "No documents loaded from PDFs",
            "documents_loaded": 0,
            "chunks_created": 0,
            "total_documents": vs.collection.count()
        }
    
    # Chunk documents
    chunks = split_documents(pdf_documents, max_characters=1000, overlap=50)
    
    # Generate embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_mgr.generate_embeddings(chunk_texts)
    
    # Add to vector store
    vs.add_documents(chunks, embeddings)
    
    total_count = vs.collection.count()
    
    return {
        "success": True,
        "message": "Documents processed and added successfully",
        "documents_loaded": len(pdf_documents),
        "chunks_created": len(chunks),
        "total_documents": total_count
    }

# Serve frontend files
@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend files not found"}

@app.get("/style.css")
async def get_css():
    """Serve CSS file."""
    from fastapi.responses import Response
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(
            content=content,
            media_type="text/css",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js")
async def get_js():
    """Serve JavaScript file."""
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve images from the public/images directory."""
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=403, detail="Access denied")
    
    image_path = PUBLIC_DIR / "images" / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine media type based on file extension
    media_type = "image/jpeg"
    if filename.lower().endswith('.png'):
        media_type = "image/png"
    elif filename.lower().endswith('.gif'):
        media_type = "image/gif"
    elif filename.lower().endswith('.svg'):
        media_type = "image/svg+xml"
    
    return FileResponse(str(image_path), media_type=media_type)

# API Routes
@app.post("/api/chat")
async def chat(query: Dict[str, str]) -> Dict[str, str]:
    """Handle chat queries using RAG system."""
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    user_query = query["query"].strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        rag = get_rag()
        answer = rag.query(user_query, top_k=3)
        
        return {
            "answer": answer,
            "query": user_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_system_loaded": rag_system is not None
    }

@app.post("/api/generate-quiz")
async def generate_quiz(request: Dict[str, Any]):
    """Generate a quiz based on a query using the RAG system."""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        num_questions = request.get("num_questions", 5)
        if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 20:
            num_questions = 5
        
        difficulty = request.get("difficulty", "medium")
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "medium"
        
        question_types = request.get("question_types", "all")
        if question_types == "all":
            q_types = ['multiple_choice', 'true_false', 'short_answer']
        else:
            q_types = [q.strip() for q in question_types.split(',')]
        
        rag = get_rag()
        quiz_gen = get_quiz_generator()
        
        # Generate quiz
        quiz = quiz_gen.generate_quiz_from_query(
            rag_system=rag,
            query=query,
            num_questions=num_questions,
            question_types=q_types,
            difficulty=difficulty,
            top_k=3
        )
        
        return {
            "success": True,
            "quiz": quiz
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document to the server.
    Only saves the file - does NOT process or create chunks/embeddings.
    Use 'Load Latest PDF' button to process the uploaded PDF."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = PDF_DIR / file.filename
        
        # Check if file already exists
        file_exists = file_path.exists()
        
        # Save the file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size for response
        file_size = file_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        print(f"📄 File uploaded: {file_path} ({size_str})")
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully. Click 'Load Latest PDF' to process it.",
            "filename": file.filename,
            "size": size_str,
            "size_bytes": file_size,
            "file_exists": file_exists,
            "uploaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/api/load-all")
async def load_all_documents():
    """Load the latest PDF document from data/pdf directory into a separate collection.
    Creates a new collection specifically for this PDF and processes it independently."""
    try:
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        
        if not pdf_files:
            return {
                "success": False,
                "message": "No PDF files found in data/pdf directory",
                "files_found": 0
            }
        
        # Find the latest PDF file (by modification time)
        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
        latest_pdf_mod_time = latest_pdf.stat().st_mtime
        latest_mod_time_str = datetime.fromtimestamp(latest_pdf_mod_time).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"📄 Latest PDF file: {latest_pdf.name} (modified: {latest_mod_time_str})")
        
        # Process the latest PDF into its own separate collection
        result = process_pdf_to_separate_collection(latest_pdf)
        
        if result["success"]:
            result["files_processed"] = 1
            result["files"] = [latest_pdf.name]
            result["latest_file"] = latest_pdf.name
            result["file_modified"] = latest_mod_time_str
            result["message"] = f"PDF '{latest_pdf.name}' processed into collection '{result['collection_name']}' with {result['chunks_created']} chunks"
        
        # Reset RAG system so it sees the new documents
        reset_rag_system()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading latest document: {str(e)}")

def retrieve_relevant_chunks_from_json(json_file_path: Path = None, top_k: int = 10) -> Dict:
    """Read JSON file(s), extract text, and retrieve most relevant chunks from vector store.
    
    Args:
        json_file_path: Path to specific JSON file. If None, processes all JSON files in Extracted_texts folder.
        top_k: Number of top chunks to retrieve for each JSON file.
    
    Returns:
        Dictionary with retrieval results
    """
    embedding_mgr = get_embedding_manager()
    vs = get_vector_store()
    retriever = RAGRetrieval(vs, embedding_mgr)
    
    # Get JSON files to process
    if json_file_path is None:
        json_files = list(EXTRACTED_TEXTS_DIR.glob("*.json"))
    else:
        json_files = [json_file_path] if json_file_path.exists() else []
    
    if not json_files:
        return {
            "success": False,
            "message": "No JSON files found",
            "json_files_processed": 0,
            "total_chunks_retrieved": 0,
            "results": []
        }
    
    all_results = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract the full_text from the JSON structure
            if 'extracted_text' in data and 'full_text' in data['extracted_text']:
                query_text = data['extracted_text']['full_text']
                source_file = data.get('source_file', json_file.name)
                
                # Use the text as a query to retrieve relevant chunks
                retrieved_chunks = retriever.retrieve(query_text, top_k=top_k, score_threshold=0.0)
                
                # Format results
                result = {
                    "json_file": json_file.name,
                    "source_file": source_file,
                    "query_text_length": len(query_text),
                    "query_preview": query_text[:200] + "..." if len(query_text) > 200 else query_text,
                    "chunks_retrieved": len(retrieved_chunks),
                    "chunks": [
                        {
                            "rank": chunk['rank'],
                            "score": round(chunk['score'], 4),
                            "document_preview": chunk['document'][:300] + "..." if len(chunk['document']) > 300 else chunk['document'],
                            "document_length": len(chunk['document']),
                            "metadata": chunk['metadata']
                        }
                        for chunk in retrieved_chunks
                    ]
                }
                all_results.append(result)
            else:
                print(f"⚠️ Warning: No 'extracted_text.full_text' found in {json_file.name}")
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            all_results.append({
                "json_file": json_file.name,
                "error": str(e),
                "chunks_retrieved": 0
            })
            continue
    
    total_chunks = sum(result.get('chunks_retrieved', 0) for result in all_results)
    
    return {
        "success": True,
        "message": "Relevant chunks retrieved successfully",
        "json_files_processed": len(json_files),
        "total_chunks_retrieved": total_chunks,
        "results": all_results
    }

@app.post("/api/retrieve-from-json")
async def retrieve_from_json(
    json_filename: str = Query(None, description="Optional filename of specific JSON file to process"),
    top_k: int = Query(10, description="Number of top chunks to retrieve")
):
    """Retrieve most relevant chunks from vector store based on JSON file text.
    
    If json_filename is provided, processes that specific file. Otherwise processes all JSON files.
    """
    try:
        if json_filename:
            json_file_path = EXTRACTED_TEXTS_DIR / json_filename
            if not json_file_path.exists():
                raise HTTPException(status_code=404, detail=f"JSON file not found: {json_filename}")
            result = retrieve_relevant_chunks_from_json(json_file_path, top_k=top_k)
        else:
            result = retrieve_relevant_chunks_from_json(top_k=top_k)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")



@app.get("/api/check-chunks")
async def check_chunks(
    query: str = Query(..., description="Text query to retrieve relevant chunks"),
    top_k: int = Query(10, description="Number of top chunks to retrieve (1-50)", ge=1, le=50),
    score_threshold: float = Query(0.0, description="Minimum similarity score (0.0 to 1.0)", ge=0.0, le=1.0)
):
    """Check and retrieve chunks from vector store based on a query text.
    
    This endpoint allows you to see what chunks are being retrieved for any query.
    Useful for debugging and understanding what content is being used.
    
    Example usage:
    GET /api/check-chunks?query=machine%20learning&top_k=5
    """
    try:
        rag = get_rag()
        vs = rag.vector_store
        
        # Check document count
        doc_count = vs.collection.count()
        if doc_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Vector store is empty. Please upload documents first."
            )
        
        # Retrieve chunks
        retrieved_chunks = rag.retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        
        # Format response with detailed chunk information
        chunks_info = []
        for chunk in retrieved_chunks:
            chunks_info.append({
                "rank": chunk.get("rank", 0),
                "score": round(chunk.get("score", 0), 4),
                "distance": round(chunk.get("distance", 0), 4),
                "document_id": chunk.get("id", ""),
                "document_length": len(chunk.get("document", "")),
                "document_preview": chunk.get("document", "")[:500] + "..." if len(chunk.get("document", "")) > 500 else chunk.get("document", ""),
                "document_full": chunk.get("document", ""),  # Full document text
                "metadata": chunk.get("metadata", {})
            })
        
        return {
            "success": True,
            "query": query,
            "query_length": len(query),
            "vector_store_total_documents": doc_count,
            "chunks_requested": top_k,
            "chunks_retrieved": len(retrieved_chunks),
            "score_threshold": score_threshold,
            "chunks": chunks_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

@app.get("/api/document-count")
async def get_document_count():
    """Get the number of documents in the vector store."""
    try:
        vs = get_vector_store()
        count = vs.collection.count()
        return {
            "document_count": count,
            "vector_store_path": vs.persist_directory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document count: {str(e)}")

@app.get("/api/pdfs")
async def list_pdfs():
    """List all PDF files in the data/pdf directory."""
    try:
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        pdf_list = []
        for pdf_file in pdf_files:
            # Get file size
            file_size = pdf_file.stat().st_size
            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            
            # Check if there's a collection for this PDF
            collection_name = sanitize_collection_name(pdf_file.name)
            has_collection = False
            collection_count = 0
            try:
                persist_dir = DATA_DIR / "Vector_Store"
                test_vs = VectorStoreDB(collection_name=collection_name, persist_directory=str(persist_dir))
                collection_count = test_vs.collection.count()
                has_collection = collection_count > 0
            except:
                pass
            
            pdf_list.append({
                "filename": pdf_file.name,
                "size": size_str,
                "size_bytes": file_size,
                "path": str(pdf_file.relative_to(BASE_DIR)),
                "collection_name": collection_name if has_collection else None,
                "has_collection": has_collection,
                "chunks_in_collection": collection_count
            })
        
        # Sort by filename
        pdf_list.sort(key=lambda x: x["filename"])
        
        return {
            "pdfs": pdf_list,
            "count": len(pdf_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")

@app.get("/api/collections")
async def list_collections():
    """List all collections in the vector store."""
    try:
        import chromadb
        persist_dir = DATA_DIR / "Vector_Store"
        client = chromadb.PersistentClient(path=str(persist_dir))
        collections = client.list_collections()
        
        collection_list = []
        for collection in collections:
            collection_list.append({
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            })
        
        return {
            "collections": collection_list,
            "count": len(collection_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    """Delete a PDF file and its associated chunks/collection from the vector store."""
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=403, detail="Access denied")
    
    file_path = PDF_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    if not file_path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="File is not a PDF")
    
    try:
        # Get document name for deleting chunks and collection
        document_name = file_path.stem
        collection_name = sanitize_collection_name(filename)
        
        deleted_chunks = 0
        collection_deleted = False
        
        # Try to delete from default collection first
        try:
            vs = get_vector_store()
            deleted_chunks = vs.delete_chunks_by_document(document_name=document_name, document_filename=filename)
        except Exception as e:
            print(f"⚠️ Could not delete from default collection: {e}")
        
        # Try to delete the separate collection if it exists
        try:
            import chromadb
            persist_dir = DATA_DIR / "Vector_Store"
            client = chromadb.PersistentClient(path=str(persist_dir))
            
            # Check if collection exists
            try:
                collection = client.get_collection(name=collection_name)
                deleted_chunks = collection.count()
                client.delete_collection(name=collection_name)
                collection_deleted = True
                print(f"🗑️ Deleted collection '{collection_name}' with {deleted_chunks} chunks")
            except Exception:
                # Collection doesn't exist, that's fine
                pass
        except Exception as e:
            print(f"⚠️ Could not delete collection '{collection_name}': {e}")
        
        # Delete the PDF file
        file_path.unlink()
        print(f"🗑️ Deleted PDF file: {filename}")
        
        # Reset RAG system if it was loaded
        reset_rag_system()
        
        return {
            "success": True,
            "message": f"PDF '{filename}' and {deleted_chunks} associated chunks deleted successfully",
            "filename": filename,
            "chunks_deleted": deleted_chunks,
            "collection_deleted": collection_deleted,
            "collection_name": collection_name if collection_deleted else None
        }
    except Exception as e:
        print(f"❌ Error deleting PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.get("/data/pdf/{filename}")
async def serve_pdf(filename: str):
    """Serve PDF files from the data/pdf directory."""
    # Security: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=403, detail="Access denied")
    
    file_path = PDF_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    if not file_path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="File is not a PDF")
    
    # Serve PDF with proper headers for iframe embedding
    return FileResponse(
        path=str(file_path),
        media_type='application/pdf',
        filename=filename,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',  # inline allows viewing in browser/iframe
            "X-Content-Type-Options": "nosniff"
        }
    )


if __name__ == "__main__":
    import uvicorn
    # NOTE: When running with reload=True, Uvicorn's file watcher will restart the server
    # whenever it detects changes under the watched directories. Since this repo contains a
    # local venv at `RAG/`, package installs can modify `RAG/lib/.../site-packages/*` and
    # trigger noisy reloads. We scope reload watching to the app code and exclude the venv.
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        reload_dirs=[str(base_dir)],
        reload_excludes=[
            "RAG/*",
            "RAG/**",
            ".venv/*",
            ".venv/**",
            "**/site-packages/**",
        ],
    )

