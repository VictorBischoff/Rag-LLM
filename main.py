import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlx.core as mx
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

class OptimizedRAGSystem:
    """Optimized RAG system using MLX best practices."""
    
    def __init__(self, 
                 pdf_path: str = "./test.pdf",
                 model_id: str = "mlx-community/granite-4.0-h-tiny-6bit-MLX",
                 cache_dir: str = "./cache",
                 chunk_size: int = 1500,
                 chunk_overlap: int = 100,
                 max_tokens: int = 500,
                 temperature: float = 0.1):
        """Initialize the RAG system with optimized settings."""
        self.pdf_path = pdf_path
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize components
        self.documents = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        
        # Performance tracking
        self.timing_stats = {}
        
    def _time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        class TimingContext:
            def __init__(self, parent, name):
                self.parent = parent
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time is not None:
                    elapsed = time.perf_counter() - self.start_time
                    self.parent.timing_stats[self.name] = elapsed
                    print(f"⏱️  {self.name}: {elapsed:.2f}s")
        
        return TimingContext(self, operation_name)
    
    def _load_documents(self) -> List[Document]:
        """Load and chunk documents with caching."""
        cache_file = self.cache_dir / "documents.pkl"
        
        if cache_file.exists():
            print("📁 Loading documents from cache...")
            with self._time_operation("Document Loading (Cached)"):
                with open(cache_file, 'rb') as f:
                    documents = pickle.load(f)
                print(f"✅ Loaded {len(documents)} cached document chunks")
                return documents
        
        print("📄 Loading and processing PDF...")
        with self._time_operation("Document Processing"):
            # Load PDF
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            document = PyMuPDFLoader(self.pdf_path).load()
            if not document:
                raise RuntimeError("No documents loaded! Check the PDF path.")
            
            print(f"📊 Loaded {len(document)} document(s)")
            print(f"📏 First document length: {len(document[0].page_content)} chars")
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            documents = splitter.split_documents(document)
            
            if not documents:
                raise RuntimeError("No chunks created! Adjust your splitter settings.")
            
            print(f"✂️  Split into {len(documents)} chunks")
            
            # Cache the processed documents
            with open(cache_file, 'wb') as f:
                pickle.dump(documents, f)
            print(f"💾 Cached documents to {cache_file}")
        
        return documents
    
    def _setup_embeddings_and_vectorstore(self):
        """Setup embeddings and vector store with caching."""
        vectorstore_cache = self.cache_dir / "vectorstore.pkl"
        
        if vectorstore_cache.exists():
            print("🔍 Loading vector store from cache...")
            with self._time_operation("Vector Store Loading (Cached)"):
                with open(vectorstore_cache, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                print("✅ Vector store loaded from cache")
                return
        
        print("🧠 Setting up embeddings and vector store...")
        with self._time_operation("Embeddings & Vector Store Setup"):
            # Use optimized embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32  # Optimize batch size
                },
            )
            
            # Create vector store
            if self.documents is None:
                raise RuntimeError("Documents not loaded. Call _load_documents() first.")
            self.vectorstore = FAISS.from_documents(self.documents, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Cache the vector store
            with open(vectorstore_cache, 'wb') as f:
                pickle.dump(self.vectorstore, f)
            print(f"💾 Cached vector store to {vectorstore_cache}")
    
    def _setup_llm_and_chain(self):
        """Setup MLX LLM with optimized parameters."""
        print("🤖 Loading MLX model...")
        with self._time_operation("MLX Model Loading"):
            # Optimize MLX model loading
            self.llm = MLXPipeline.from_model_id(
                self.model_id,
                pipeline_kwargs={
                    "max_tokens": self.max_tokens,
                    "temp": self.temperature,
                    "repetition_penalty": 1.1,  # Add repetition penalty
                    "repetition_context_size": 20,
                },
            )
        
        print("🔗 Setting up RAG chain...")
        with self._time_operation("Chain Setup"):
            # Optimized prompt template
            template = """You are a helpful assistant that answers questions based on the provided context. 

Instructions:
- Answer the question clearly and concisely
- Use only the information from the provided context
- If the context doesn't contain enough information, say "I don't have enough information in the provided context to answer this question"
- Format your response in a clean, readable way
- Avoid repeating raw text from the context
- Be specific and cite relevant parts when possible

Question: {input}

Context: {context}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            doc_chain = create_stuff_documents_chain(self.llm, prompt)
            self.chain = create_retrieval_chain(self.retriever, doc_chain)
    
    def initialize(self):
        """Initialize the complete RAG system."""
        print("🚀 Initializing Optimized RAG System...")
        print("=" * 50)
        
        # Load documents
        self.documents = self._load_documents()
        
        # Setup embeddings and vector store
        self._setup_embeddings_and_vectorstore()
        
        # Setup LLM and chain
        self._setup_llm_and_chain()
        
        print("=" * 50)
        print("✅ RAG System initialized successfully!")
        print(f"📊 Performance Summary:")
        for operation, time_taken in self.timing_stats.items():
            print(f"   {operation}: {time_taken:.2f}s")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with timing."""
        if not self.chain:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        print(f"\n❓ Question: {question}")
        with self._time_operation("Query Processing"):
            response = self.chain.invoke({"input": question})
        
        print(f"✅ Answer: {response['answer']}")
        return response
    
    def interactive_mode(self):
        """Run interactive query mode."""
        if not self.chain:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        print("\n🎯 Interactive Mode - Ask questions about your document!")
        print("Type 'quit' to exit, 'stats' to see performance stats")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'stats':
                    self._print_stats()
                    continue
                elif not question:
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _print_stats(self):
        """Print performance statistics."""
        print("\n📊 Performance Statistics:")
        print("-" * 30)
        for operation, time_taken in self.timing_stats.items():
            print(f"{operation}: {time_taken:.2f}s")


def main():
    """Main function to run the optimized RAG system."""
    try:
        # Initialize the RAG system
        rag = OptimizedRAGSystem(
            pdf_path="./test.pdf",
            model_id="mlx-community/granite-4.0-h-tiny-6bit-MLX",
            cache_dir="./cache",
            chunk_size=1500,
            chunk_overlap=100,
            max_tokens=500,
            temperature=0.1
        )
        
        # Initialize the system
        rag.initialize()
        
        # Run interactive mode
        rag.interactive_mode()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())