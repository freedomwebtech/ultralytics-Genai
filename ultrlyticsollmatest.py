import os
import gradio as gr
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
from ultralytics import YOLO

# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = ""
model = YOLO("best.pt")  # Use the trained YOLO model (e.g., detecting Arduino Uno)
names = model.names
# Load Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

def process_input(image_path, text):
    if not image_path or not text:
        return "Please provide both an image and a text prompt."

    # Run YOLO object detection
    results = model(image_path)  # Perform inference on the input image

    if results[0].boxes is not None:
        # Get the boxes, class IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores

        # Try to get track IDs, handle the case if they are not available
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None] * len(boxes)

        detected_objects = []
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            detected_objects.append(c)

        # Combine detected objects as part of the query to Gemini
        detected_objects_text = ', '.join(detected_objects)
        enhanced_text = f"Objects detected: {detected_objects_text}. {text}"

    else:
        # If no objects detected, use the original text
        enhanced_text = text

    # Prepare the message for Gemini
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text=enhanced_text),
            ImageBlock(path=image_path, image_mimetype="image/jpeg"),  # Ensure MIME type is set
        ],
    )

    try:
        # Send the message to Gemini
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
