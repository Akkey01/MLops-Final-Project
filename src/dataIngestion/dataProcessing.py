from dataChunking import chunkText
from textProcessor import textProcessor
# from embedding import embedding
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import os

def post_process_text(raw_text, source_type=""):
        """
        Common text cleaning and chunking logic for all types of input.
        """
    
        processed_text = textProcessor(raw_text)
        chunks = chunkText(processed_text)

        # print(f"\nChunks from {source_type} file:")
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nChunk {idx+1}:\n{chunk}")

        return chunks  
    
chunks=post_process_text("Intelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises Intelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for EnterprisesIntelligent Multimedia Processing (IMP) for Enterprises","document")

model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding(texts: list[str]) -> list[list[float]]:
    # returns one vector per text
    return model.encode(texts, show_progress_bar=True)



embeddings = embedding(chunks)
print(embeddings[0])  # first chunk’s vector
import numpy as np
import faiss

# ──────────────────────────────────────────────────────────────────────────────
# 1) Build a FAISS index from your embeddings
# ──────────────────────────────────────────────────────────────────────────────

# Stack your list of embedding vectors into a single NumPy array
# embeddings is List[List[float]] from your embedding(chunks) call
emb_array = np.vstack(embeddings).astype("float32")

# Determine dimensionality
dim = emb_array.shape[1]          # e.g. 384 for all-MiniLM-L6-v2

# Create a simple flat (L2) index and add your vectors
index = faiss.IndexFlatL2(dim)
index.add(emb_array)

print(f"✅ Indexed {index.ntotal} chunks (dimension={dim})")

# (Optional) persist to disk so you don’t re-embed next time
faiss.write_index(index, "imp_faiss.index")

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# point Chroma at a folder on disk
client = chromadb.Client(
  Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db/chroma_imp"
  )
)
# use the same model you used to embed your chunks
echf = embedding_functions.SentenceTransformerEmbeddingFunction(
  model_name="all-MiniLM-L6-v2"
)

col = client.get_or_create_collection(
  name="imp_chunks",
  embedding_function=echf
)
