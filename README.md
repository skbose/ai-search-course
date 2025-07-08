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
