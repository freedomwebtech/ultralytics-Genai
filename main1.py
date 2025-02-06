import os
import gradio as gr
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
from ultralytics import YOLO
# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkBApCCKznY1OjcSqcvCTMj3ZFXiTR4UI"
model = YOLO("best.pt")  # Use the trained YOLO model (e.g., detecting Arduino Uno)
names = model.names
# Load Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")


if results[0].boxes is not None:
    # Get the boxes, class IDs, and confidences
    boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
    class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
    confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores

    # Try to get track IDs, handle the case if they are not available
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None] * len(boxes)

    for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
        c = names[class_id]
        
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