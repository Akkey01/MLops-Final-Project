import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def textProcessor(raw_text):
    # Text cleaning and standardization
    processed_text = raw_text.lower()
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = re.sub(r'[^a-z0-9\s.,!?]', '', processed_text)
    
    # # Generate embedding
    # embedding = model.encode(processed_text)

    return processed_text


# text = textProcessor("Intelligent Multimedia Processing (IMP) for Enterprises")

# print("Processed Text:", text)


