import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    persist_directory="./chroma_db",  # where to store on-disk data
))

collection = client.get_or_create_collection(
    name="my_data",
    metadata={"description": "RAG index of PDF/audio chunks"}
)

# prepare records
ids     = [c["id"]     for c in chunks]
metas   = [ {"source":c["source"]} for c in chunks ]
vectors = embeddings

collection.upsert(
    ids=ids,
    metadatas=metas,
    embeddings=vectors,
    documents=texts
)

# persist to disk
client.persist()
