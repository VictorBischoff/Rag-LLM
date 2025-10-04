"""
Unit tests for OptimizedRAGSystem in main.py.

These tests mock all external dependencies (PyMuPDFLoader, HuggingFaceEmbeddings,
FAISS, MLXPipeline) to run without GPU or network access.
"""

import pickle
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from main import OptimizedRAGSystem
from tests.conftest import create_test_pdf, assert_timing_stats_present


class PickleableRetriever:
    """Simple retriever stub that supports pickling."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - helper for mocks
        raise NotImplementedError


class PickleableVectorStore:
    """Vector store stub that can be serialized with pickle."""

    def __init__(self):
        self._retriever = PickleableRetriever()

    def as_retriever(self, **_search_kwargs):
        return self._retriever


@pytest.mark.unit
class TestOptimizedRAGSystem:
    """Test cases for OptimizedRAGSystem class."""

    def test_init_with_single_pdf_path(self, temp_dir):
        """Test initialization with single PDF path (backward compatibility)."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=str(pdf_path), cache_dir=str(temp_dir))

        assert rag.pdf_paths == [str(pdf_path)]
        assert rag.model_id == "mlx-community/granite-4.0-h-tiny-4bit"
        assert rag.cache_dir == temp_dir
        assert rag.chunk_size == 1500
        assert rag.chunk_overlap == 100
        assert rag.max_tokens == 500
        assert rag.temperature == 0.1
        assert rag.documents is None
        assert rag.vectorstore is None
        assert rag.retriever is None
        assert rag.llm is None
        assert rag.chain is None
        assert rag.timing_stats == {}

    def test_init_with_list_pdf_paths(self, temp_dir):
        """Test initialization with list of PDF paths."""
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1)
        create_test_pdf(pdf2)

        rag = OptimizedRAGSystem(
            pdf_paths=[str(pdf1), str(pdf2)], cache_dir=str(temp_dir)
        )

        assert rag.pdf_paths == [str(pdf1), str(pdf2)]

    def test_init_with_none_pdf_paths(self, temp_dir):
        """Test initialization with None PDF paths (uses default)."""
        rag = OptimizedRAGSystem(pdf_paths=None, cache_dir=str(temp_dir))

        assert rag.pdf_paths == ["./test.pdf"]

    def test_init_creates_cache_dir(self, temp_dir):
        """Test that cache directory is created during initialization."""
        cache_dir = temp_dir / "new_cache"
        assert not cache_dir.exists()

        OptimizedRAGSystem(cache_dir=str(cache_dir))

        assert cache_dir.exists()

    def test_generate_cache_key_stability(self, temp_dir):
        """Test that cache key generation is stable and consistent."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        # Generate cache key multiple times
        key1 = rag._generate_cache_key()
        key2 = rag._generate_cache_key()

        assert key1 == key2
        assert len(key1) == 16  # MD5 hash truncated to 16 chars
        assert isinstance(key1, str)

    def test_generate_cache_key_with_missing_file(self, temp_dir):
        """Test cache key generation with missing PDF file."""
        missing_pdf = temp_dir / "missing.pdf"

        rag = OptimizedRAGSystem(pdf_paths=[str(missing_pdf)], cache_dir=str(temp_dir))

        key = rag._generate_cache_key()
        assert len(key) == 16
        assert isinstance(key, str)

    def test_generate_cache_key_multiple_files(self, temp_dir):
        """Test cache key generation with multiple PDF files."""
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1, "Content 1")
        create_test_pdf(pdf2, "Content 2")

        rag = OptimizedRAGSystem(
            pdf_paths=[str(pdf1), str(pdf2)], cache_dir=str(temp_dir)
        )

        key = rag._generate_cache_key()
        assert len(key) == 16
        assert isinstance(key, str)

    @patch("main.PyMuPDFLoader")
    def test_load_documents_success(self, mock_loader, temp_dir, sample_documents):
        """Test successful document loading."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mock
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        documents = rag._load_documents()

        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        mock_loader.assert_called_once_with(str(pdf_path))
        mock_loader_instance.load.assert_called_once()

    @patch("main.PyMuPDFLoader")
    def test_load_documents_with_caching(self, mock_loader, temp_dir, sample_documents):
        """Test document loading with caching."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mock
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        # First load - should call the loader
        documents1 = rag._load_documents()
        assert len(documents1) == 3
        assert mock_loader_instance.load.call_count == 1

        # Second load - should use cache
        documents2 = rag._load_documents()
        assert len(documents2) == 3
        assert mock_loader_instance.load.call_count == 1  # Still 1, not 2

    def test_load_documents_missing_pdf(self, temp_dir):
        """Test document loading with missing PDF file."""
        missing_pdf = temp_dir / "missing.pdf"

        rag = OptimizedRAGSystem(pdf_paths=[str(missing_pdf)], cache_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            rag._load_documents()

    @patch("main.PyMuPDFLoader")
    def test_load_documents_empty_content(self, mock_loader, temp_dir):
        """Test document loading with empty content."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mock to return empty list
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        with pytest.raises(
            RuntimeError, match="No documents loaded from any PDF files"
        ):
            rag._load_documents()

    @patch("main.PyMuPDFLoader")
    @patch("main.RecursiveCharacterTextSplitter")
    def test_load_documents_chunking_failure(
        self, mock_splitter, mock_loader, temp_dir, sample_documents
    ):
        """Test document loading when chunking fails."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = []
        mock_splitter.return_value = mock_splitter_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="No chunks created"):
            rag._load_documents()

    @patch("main.HuggingFaceEmbeddings")
    @patch("main.FAISS")
    @patch("builtins.open", create=True)
    def test_setup_embeddings_and_vectorstore_success(
        self, mock_open, mock_faiss, mock_embeddings, temp_dir, sample_documents
    ):
        """Test successful embeddings and vectorstore setup."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore

        # Mock file operations to avoid pickle issues
        mock_open.return_value.__enter__ = Mock()
        mock_open.return_value.__exit__ = Mock()

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))
        rag.documents = sample_documents

        rag._setup_embeddings_and_vectorstore()

        assert rag.vectorstore == mock_vectorstore
        assert rag.retriever == mock_retriever
        mock_embeddings.assert_called_once()
        mock_faiss.from_documents.assert_called_once_with(
            sample_documents, mock_embeddings_instance
        )

    @patch("main.HuggingFaceEmbeddings")
    @patch("main.FAISS")
    def test_setup_embeddings_with_caching(
        self, mock_faiss, mock_embeddings, temp_dir, sample_documents
    ):
        """Test embeddings setup with caching."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))
        rag.documents = sample_documents

        # Generate cache key and create cache file
        cache_key = rag._generate_cache_key()
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        vectorstore_cache = cache_dir / f"vectorstore_{cache_key}.pkl"

        # Create a mock vectorstore to pickle
        mock_cached_vectorstore = PickleableVectorStore()
        mock_cached_retriever = mock_cached_vectorstore.as_retriever()

        # Write the cache file
        with open(vectorstore_cache, "wb") as f:
            pickle.dump(mock_cached_vectorstore, f)

        # First setup - should load from cache
        rag._setup_embeddings_and_vectorstore()
        assert rag.vectorstore == mock_cached_vectorstore
        assert rag.retriever == mock_cached_retriever
        # Should not call FAISS.from_documents since we loaded from cache
        assert mock_faiss.from_documents.call_count == 0

    def test_setup_embeddings_no_documents(self, temp_dir):
        """Test embeddings setup without documents loaded."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))
        # Don't set rag.documents

        with pytest.raises(RuntimeError, match="Documents not loaded"):
            rag._setup_embeddings_and_vectorstore()

    @patch("main.MLXPipeline")
    @patch("main.create_stuff_documents_chain")
    @patch("main.create_retrieval_chain")
    def test_setup_llm_and_chain_success(
        self, mock_retrieval_chain, mock_doc_chain, mock_mlx_pipeline, temp_dir
    ):
        """Test successful LLM and chain setup."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mocks
        mock_llm = Mock()
        mock_mlx_pipeline.from_model_id.return_value = mock_llm

        mock_doc_chain_instance = Mock()
        mock_doc_chain.return_value = mock_doc_chain_instance

        mock_retrieval_chain_instance = Mock()
        mock_retrieval_chain.return_value = mock_retrieval_chain_instance

        mock_retriever = Mock()

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))
        rag.retriever = mock_retriever

        rag._setup_llm_and_chain()

        assert rag.llm == mock_llm
        assert rag.chain == mock_retrieval_chain_instance
        mock_mlx_pipeline.from_model_id.assert_called_once()
        mock_doc_chain.assert_called_once()
        mock_retrieval_chain.assert_called_once_with(
            mock_retriever, mock_doc_chain_instance
        )

    @patch("main.PyMuPDFLoader")
    @patch("main.HuggingFaceEmbeddings")
    @patch("main.FAISS")
    @patch("main.MLXPipeline")
    @patch("main.create_stuff_documents_chain")
    @patch("main.create_retrieval_chain")
    def test_initialize_success(
        self,
        mock_retrieval_chain,
        mock_doc_chain,
        mock_mlx_pipeline,
        mock_faiss,
        mock_embeddings,
        mock_loader,
        temp_dir,
        sample_documents,
    ):
        """Test successful system initialization."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup all mocks
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore

        mock_llm = Mock()
        mock_mlx_pipeline.from_model_id.return_value = mock_llm

        mock_doc_chain_instance = Mock()
        mock_doc_chain.return_value = mock_doc_chain_instance

        mock_retrieval_chain_instance = Mock()
        mock_retrieval_chain.return_value = mock_retrieval_chain_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        rag.initialize()

        # Verify all components are set
        assert rag.documents == sample_documents
        assert rag.vectorstore == mock_vectorstore
        assert rag.retriever == mock_retriever
        assert rag.llm == mock_llm
        assert rag.chain == mock_retrieval_chain_instance

        # Verify timing stats are recorded
        expected_operations = [
            "Document Processing",
            "Embeddings & Vector Store Setup",
            "MLX Model Loading",
            "Chain Setup",
        ]
        assert_timing_stats_present(rag, expected_operations)

    def test_query_before_initialization(self, temp_dir):
        """Test that query raises error before initialization."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="RAG system not initialized"):
            rag.query("What is this document about?")

    def test_query_success(self, stubbed_rag_system):
        """Test successful query execution."""
        question = "What is this document about?"

        result = stubbed_rag_system.query(question)

        assert "answer" in result
        assert "context" in result
        assert result["answer"] == "Mock answer"
        assert result["context"] == "Mock context"

        # Verify timing stats are recorded
        assert "Query Processing" in stubbed_rag_system.timing_stats

    def test_query_timing_stats(self, stubbed_rag_system, mock_time):
        """Test that query records timing statistics."""
        question = "What is this document about?"

        stubbed_rag_system.query(question)

        assert "Query Processing" in stubbed_rag_system.timing_stats
        assert isinstance(
            stubbed_rag_system.timing_stats["Query Processing"], (int, float)
        )

    def test_interactive_mode_before_initialization(self, temp_dir):
        """Test that interactive mode raises error before initialization."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        with pytest.raises(RuntimeError, match="RAG system not initialized"):
            rag.interactive_mode()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_quit_command(
        self, mock_print, mock_input, stubbed_rag_system
    ):
        """Test interactive mode with quit command."""
        mock_input.side_effect = ["quit"]

        stubbed_rag_system.interactive_mode()

        mock_input.assert_called_once()
        # Should print goodbye message
        mock_print.assert_any_call("👋 Goodbye!")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_stats_command(
        self, mock_print, mock_input, stubbed_rag_system
    ):
        """Test interactive mode with stats command."""
        # Add some timing stats
        stubbed_rag_system.timing_stats = {"Test Operation": 1.5}
        mock_input.side_effect = ["stats", "quit"]

        stubbed_rag_system.interactive_mode()

        # Should print stats
        mock_print.assert_any_call("📊 Performance Statistics:")
        mock_print.assert_any_call("Test Operation: 1.5s")

    @patch("builtins.input")
    def test_interactive_mode_query_processing(self, mock_input, stubbed_rag_system):
        """Test interactive mode processes queries correctly."""
        mock_input.side_effect = ["What is this about?", "quit"]

        stubbed_rag_system.interactive_mode()

        # Verify the chain was invoked with the question
        stubbed_rag_system.chain.invoke.assert_called_with(
            {"input": "What is this about?"}
        )

    def test_time_operation_context_manager(self, temp_dir, mock_time):
        """Test the _time_operation context manager."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        with rag._time_operation("Test Operation") as timer:
            assert timer.parent == rag
            assert timer.name == "Test Operation"
            assert timer.start_time is not None

        # Verify timing was recorded
        assert "Test Operation" in rag.timing_stats
        assert rag.timing_stats["Test Operation"] == 1.5  # From mock_time fixture

    def test_logger_setup(self, temp_dir):
        """Test that logger is properly set up."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        assert rag.logger is not None
        assert rag.logger.name.startswith("RAGSystem_")
        assert len(rag.logger.handlers) > 0

    @patch("main.PyMuPDFLoader")
    def test_document_metadata_enhancement(self, mock_loader, temp_dir):
        """Test that document metadata is properly enhanced with source information."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Create documents without source metadata
        documents = [
            Document(page_content="Test content 1", metadata={}),
            Document(page_content="Test content 2", metadata={}),
        ]

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = documents
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        loaded_documents = rag._load_documents()

        for doc in loaded_documents:
            assert "source_file" in doc.metadata
            assert "source_path" in doc.metadata
            assert doc.metadata["source_file"] == "test.pdf"
            assert doc.metadata["source_path"] == str(pdf_path)

    @patch("main.PyMuPDFLoader")
    def test_multiple_pdf_processing(self, mock_loader, temp_dir):
        """Test processing multiple PDF files."""
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1, "Content 1")
        create_test_pdf(pdf2, "Content 2")

        # Create different documents for each PDF
        docs1 = [Document(page_content="Content 1", metadata={})]
        docs2 = [Document(page_content="Content 2", metadata={})]

        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = [docs1, docs2]
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(
            pdf_paths=[str(pdf1), str(pdf2)], cache_dir=str(temp_dir)
        )

        loaded_documents = rag._load_documents()

        assert len(loaded_documents) == 2
        assert mock_loader_instance.load.call_count == 2

        # Verify source metadata is set correctly
        assert loaded_documents[0].metadata["source_file"] == "test1.pdf"
        assert loaded_documents[1].metadata["source_file"] == "test2.pdf"

    def test_cache_key_consistency_across_instances(self, temp_dir):
        """Test that cache keys are consistent across different RAG instances."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        rag1 = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))
        rag2 = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        key1 = rag1._generate_cache_key()
        key2 = rag2._generate_cache_key()

        assert key1 == key2

    @patch("main.PyMuPDFLoader")
    def test_corrupt_cache_handling(self, mock_loader, temp_dir, sample_documents):
        """Test handling of corrupt cache files."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path)

        # Setup mock for fallback loading
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_loader_instance

        rag = OptimizedRAGSystem(pdf_paths=[str(pdf_path)], cache_dir=str(temp_dir))

        # Generate the actual cache key and create corrupt cache file
        cache_key = rag._generate_cache_key()
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / f"documents_{cache_key}.pkl"
        cache_file.write_bytes(b"corrupt data")

        # Should fall back to loading from PDF when cache is corrupt
        documents = rag._load_documents()

        assert len(documents) == 3
        mock_loader_instance.load.assert_called_once()
