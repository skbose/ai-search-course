# A Course on NLP Based Search
A repository for tracking project related to NLP search course

# Setup Instructions

## Install Tools

1. [Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation)
2. [Install Docker](https://docs.docker.com/desktop/)
3. [Install Qdrant](https://qdrant.tech/documentation/guides/installation/)

## Setup Environment

Step 1. Create conda environment
```
conda init (optional)
conda create -n ai-search python=3.10
```

Step 2. Install packages
```
pip install poetry

poetry init
poetry add gradio

poetry add langchain
poetry add langchain[openai]
poetry add langchain[anthropic]

poetry add sentence-transformers
```


## 🚀 Steps to Run the Application

### ✅ 1. Create Environment File
Copy the `.env-example` to `.env` and update the values as needed


### ✅ 2. Install Dependencies Using Poetry
poetry install

### ✅ 3. Start Qdrant Vector Store
docker run -p 6333:6333 qdrant/qdrant


### ✅ 4. Start the FastAPI App
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000


### ✅ 5. Interface URL to upload PDF files and to Generate Embeddings
we have stored 2 sample files in /data folder
http://127.0.0.1:8000/admin/file-upload


### ✅ 6. Query Using Semantic Search (User chat interface URL)
http://127.0.0.1:8000/user-chat
