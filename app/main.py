from app.ui.user.query import chat_with_model_ui
from dotenv import load_dotenv
load_dotenv()
import os

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY", "QDRANT_HOST", "QDRANT_COLLECTION"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from app.ui.admin.file_upload import get_upload_ui

app = FastAPI(
    title="AI Document Search API",
    description="PDF upload, embedding generation, and vector search",
    version="1.0.0"
)

# Admin routes for file upload
mount_gradio_app(app, get_upload_ui(), path="/admin/file-upload")

# User query application route
mount_gradio_app(app, chat_with_model_ui(), path="/user-chat")
