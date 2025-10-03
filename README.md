# RAG PDF Query API

A simple web API for uploading PDF documents and querying them using Retrieval-Augmented Generation (RAG) with MLX on Apple Silicon.

## Features

- 📄 **PDF Upload**: Upload PDF files and automatically process them
- 🤖 **RAG Queries**: Ask questions about your uploaded documents
- ⚡ **MLX Integration**: Fast inference using MLX on Apple Silicon
- 🔄 **Session Management**: Multiple document sessions with unique IDs
- 📊 **Performance Tracking**: Built-in timing and performance metrics
- 🛡️ **Error Handling**: Comprehensive error handling and validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. View API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Core Endpoints

- `POST /upload` - Upload a PDF file
- `POST /query` - Query a document
- `GET /sessions` - List active sessions
- `DELETE /sessions/{session_id}` - Delete a specific session
- `GET /health` - Health check

### Example Usage

#### Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "PDF uploaded and processed successfully",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "document_count": 45,
  "processing_time": 12.34
}
```

#### Query a Document

```bash
curl -X POST "http://localhost:8000/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the main topic of this document?",
       "session_id": "123e4567-e89b-12d3-a456-426614174000"
     }'
```

Response:
```json
{
  "answer": "The main topic of this document is...",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "processing_time": 2.15
}
```

## Using the Client Script

A Python client script is provided for easy testing:

```bash
# Upload a PDF and start interactive mode
python client_example.py test.pdf

# Or use existing sessions
python client_example.py
```

The client script provides:
- PDF upload functionality
- Interactive query mode
- Session management
- Error handling

## Session Management

The API supports multiple concurrent document sessions:

- Each uploaded PDF gets a unique session ID
- Sessions persist until explicitly deleted
- You can query any active session by providing the session_id
- If no session_id is provided, the first available session is used

## Configuration

The API uses the following default settings:

- **Model**: `mlx-community/granite-4.0-h-tiny-6bit-MLX`
- **Chunk Size**: 1500 characters
- **Chunk Overlap**: 100 characters
- **Max Tokens**: 500
- **Temperature**: 0.1

These can be modified in the `api.py` file if needed.

## Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid file type or missing parameters
- **404 Not Found**: Session not found
- **500 Internal Server Error**: Processing errors

All errors include descriptive messages to help with debugging.

## Performance

The system includes several optimizations:

- **Caching**: Document chunks and vector stores are cached
- **Batch Processing**: Optimized embedding generation
- **MLX Integration**: Fast inference on Apple Silicon
- **Timing Metrics**: Built-in performance tracking

## Development

### Running in Development Mode

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Requirements

- Python 3.8+
- Apple Silicon Mac (for MLX)
- Sufficient RAM for model loading (recommended: 8GB+)

## Troubleshooting

### Common Issues

1. **"No active sessions"**: Upload a PDF first before querying
2. **"Session not found"**: Check session ID or upload a new PDF
3. **Memory issues**: Try reducing chunk_size or max_tokens
4. **Slow performance**: Ensure you're using Apple Silicon for optimal MLX performance

### Logs

The API provides detailed logging. Check the console output for:
- Processing times
- Error messages
- Performance statistics

## License

This project uses the same license as the original RAG system.
