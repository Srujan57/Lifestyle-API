import os
import sys
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection so we start fresh each ingest run
try:
    chroma_client.delete_collection("health_docs")
except Exception:
    pass

collection = chroma_client.get_or_create_collection("health_docs")

# Chunk settings – 500 chars with 50-char overlap to preserve context across boundaries
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs_folder = "./documents"
all_chunks = []
all_ids = []
chunk_index = 0

if not os.path.isdir(docs_folder):
    print(f"Documents folder '{docs_folder}' does not exist. Creating it.")
    os.makedirs(docs_folder, exist_ok=True)

files = os.listdir(docs_folder)
if not files:
    print("No files found in documents/. Add PDF or TXT files and re-run.")
    sys.exit(0)

for filename in files:
    filepath = os.path.join(docs_folder, filename)
    text = ""

    if filename.endswith(".pdf"):
        print(f"Reading PDF: {filename}")
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() or ""

    elif filename.endswith(".txt"):
        print(f"Reading TXT: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        print(f"Skipping unsupported file: {filename}")
        continue

    if not text.strip():
        print(f"  → WARNING: No text extracted from {filename}, skipping.")
        continue

    chunks = splitter.split_text(text)
    print(f"  → {len(chunks)} chunks created")

    for chunk in chunks:
        all_chunks.append(chunk)
        all_ids.append(f"chunk_{chunk_index}")
        chunk_index += 1

if not all_chunks:
    print("No text chunks were produced. Check your documents.")
    sys.exit(0)

print(f"\nTotal chunks to embed: {len(all_chunks)}")
print("Sending to OpenAI for embedding (this may take a moment)...")

# Embed in batches of 100 to avoid hitting API limits on large document sets
BATCH_SIZE = 100
all_embeddings = []

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i : i + BATCH_SIZE]
    response = openai_client.embeddings.create(
        input=batch,
        model="text-embedding-3-small",
    )
    all_embeddings.extend([item.embedding for item in response.data])
    print(f"  Embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

# Store in ChromaDB
collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=all_ids,
)

print(f"Done! {len(all_chunks)} chunks saved to chroma_db/")
