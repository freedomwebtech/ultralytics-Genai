import os
import gradio as gr
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock

# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = ""

# Load Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

def process_input(image_path, text):
    if not image_path or not text:
        return "Please provide both an image and a text prompt."

    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text=text),
            ImageBlock(path=image_path, image_mimetype="image/jpeg"),  # Ensure MIME type is set
        ],
    )

    try:
        response = gemini_pro.chat(messages=[msg])
        return response.message.content if response.message.content else "No response from the model."
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio Interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Image(type="filepath"), gr.Textbox(label="Enter your question")],
    outputs="text",
    title="Gemini Vision Model",
    description="Upload an image and ask a question about it.",
)

iface.launch()
