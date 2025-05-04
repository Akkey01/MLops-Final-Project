# RAG-Powered Chat App

This repository contains a Streamlit-based Retrieval-Augmented Generation (RAG) app.

## Files
- `app.py`: Streamlit application code
- `Dockerfile`: Docker configuration
- `requirements.txt`: Python dependencies

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. Run Locally
	streamlit run app.py

Docker
	docker build -t rag-chat-app .

	docker run -p 8501:8501 rag-chat-app
 If you wish to use a different data source or your own folder, you can override:
 	docker run -p 8501:8501 \
  	-e DATA_DIR=/app/ami_data \
 	-v /your/local/AMI:/app/ami_data \
 	rag-chat

  


