
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import os

# def embedding(chunks):
    # # TOGETHER_API_KEY = "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"  # paste the key from your dashboard

    # # TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    # # embedder = OpenAIEmbeddings(openai_api_key=TOGETHER_API_KEY)
    # embedder = OpenAIEmbeddings(
    #     openai_api_key="cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"
    # )
    # # Batch-embed all chunks
    # # texts = [c["text"] for c in chunks]
    # embeddings = embedder.embed_documents(chunks)
    # print(embeddings)
    # # embeddings is List[List[float]]

   

   
model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding(texts: list[str]) -> list[list[float]]:
    # returns one vector per text
    return model.encode(texts, show_progress_bar=True)



embeddings = embedding(chunks)
print(embeddings[0])  # first chunk’s vector

