
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import os

# def embedding(chunks):


   
model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding(texts: list[str]) -> list[list[float]]:
    # returns one vector per text
    return model.encode(texts, show_progress_bar=True)



embeddings = embedding(chunks)
print(embeddings[0])  # first chunk’s vector

