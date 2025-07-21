import gradio as gr

def upload_file(file):
    return f"Uploaded {file.name}"

def get_upload_ui():
    return gr.Interface(fn=upload_file, inputs=gr.File(file_types=[".pdf"]), outputs="text", title="Upload PDF")
