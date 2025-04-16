import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def textProcessor(raw_text):
    # Text cleaning and standardization
    processed_text = raw_text.lower()
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = re.sub(r'[^a-z0-9\s.,!?]', '', processed_text)
    
    # Generate embedding
    embedding = model.encode(processed_text)

    return processed_text, embedding


text, vector = textProcessor("My name is Anthony Gonsalves")
print("Processed Text:", text)
print("Vector Embedding shape:", vector.shape)

