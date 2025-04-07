import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uuid

load_dotenv()

pdf_directory = "/Users/brianfedelin/Desktop/code/aparavi/agentic_rag_chatbot/knowledge/aparavi "

COLLECTION_NAME = 'aparavi_knowledge'
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

openai_client = openai.Client(
    api_key=os.getenv("OPENAI_API_KEY")
)
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

chunks = []
chunk_metadata = []
ids = []

# # Split chunks at headings
# chunker = RecursiveCharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=200,
#     separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
# )

# # Process each PDF
# for pdf_file in [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]:
#     file_path = os.path.join(pdf_directory, pdf_file)
#     print(f"Processing {file_path}")

#     # Extract text from PDF
#     text = ""
#     reader = PdfReader(file_path)
#     for page in reader.pages:
#         text += page.extract_text() + "\n"

#     # Chunk the document
#     text_chunks = chunker.split_text(text)

#     # Store chunks with metadata
#     for i, chunk in enumerate(text_chunks):
#         ids.append(str(uuid.uuid4()))
#         chunks.append(chunk)
#         chunk_metadata.append({
#             "source": pdf_file,
#             "chunk_index": i
#         })
    
#     print(pdf_file, ": ", len(text_chunks))


# qdrant_client.add(
#     collection_name=COLLECTION_NAME,
#     documents=chunks,
#     metadata=chunk_metadata,
#     ids=ids
# )

search_result = qdrant_client.query(
    collection_name="aparavi_knowledge",
    query_text="how do i log into aparavi?"
)

for response in search_result:
    print(response.metadata['document'])
    print(response.metadata['source'])

