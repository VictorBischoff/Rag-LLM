from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma  # <-- remove Chroma to avoid hnswlib
from langchain_community.vectorstores import FAISS
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# 1. Load and chunk documents
document = PyMuPDFLoader("./test.pdf").load()
print(f"Loaded {len(document)} document(s)")
if document:
    print(f"First document content preview: {repr(document[0].page_content[:500])}")
    print(f"Document length: {len(document[0].page_content)}")
    print(f"Document metadata: {document[0].metadata}")
else:
    raise RuntimeError("No documents loaded! Check the PDF path.")

documents = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100
).split_documents(document)
print(f"Split into {len(documents)} chunks")
if documents:
    print(f"First chunk preview: {documents[0].page_content[:200]}...")
else:
    raise RuntimeError("No chunks created! Adjust your splitter settings.")

# 2. Embed and index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# FAISS replaces Chroma:
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Load MLX LLM
llm = MLXPipeline.from_model_id(
    "mlx-community/granite-4.0-h-tiny-6bit-MLX",
    pipeline_kwargs={"max_tokens": 500, "temp": 0.1},
)

# 4. Define prompt and chains
template = """You are a helpful assistant that answers questions based on the provided context. 

Instructions:
- Answer the question clearly and concisely
- Use only the information from the provided context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question"
- Format your response in a clean, readable way
- Avoid repeating raw text from the context

Question: {input}

Context: {context}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)

# 5. Interactive loop
while True:
    question = input("\nEnter your question (or 'quit' to exit): ")
    if question.lower() == "quit":
        break
    response = chain.invoke({"input": question})
    print("\nAnswer:", response["answer"])