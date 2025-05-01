
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import os

# def embedding(chunks):
    # # TOGETHER_API_KEY = ""  # paste the key from your dashboard

    # # TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    # # embedder = OpenAIEmbeddings(openai_api_key=TOGETHER_API_KEY)
    # embedder = OpenAIEmbeddings(
    #     openai_api_key=""
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

