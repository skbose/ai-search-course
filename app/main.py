from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from app.ui.admin.file_upload import get_upload_ui

app = FastAPI()

# Admin routes for file upload
mount_gradio_app(app, get_upload_ui(), path="/admin/file-upload")

# User query application route
