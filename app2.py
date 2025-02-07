import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import torch
from torchvision import transforms
import torch.nn as nn
from io import BytesIO

# Custom CSS to enhance the look
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f4f9;
        }
        h1, h3, p {
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        h3 {
            font-family: 'Arial', sans-serif;
            color: #333;
            text-align: center;
        }
        .stMarkdown {
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            color: #666;
            line-height: 1.6;
        }
        .stImage {
            display: block;
            margin: 20px auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSpinner {
            margin-top: 50px;
            text-align: center;
            font-size: 18px;
            color: #4CAF50;
        }
        .footer {
            text-align: center;
            padding: 20px 0;
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Define the CNN architecture for emotion detection
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model with caching
@st.cache_resource
def load_model():
    model = EmotionCNN()
    model.load_state_dict(torch.load(r'C:\Users\Priyanka Malavade\OneDrive\Desktop\emotion_detaction\emotion_cnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Function to process the uploaded image and detect face
def detect_face(image):
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(faces) == 0:
        return image, None
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, faces

# Function to classify emotion using the trained model
def classify_emotion(face_image_rgb):
    face_image_gray = face_image_rgb.convert("L")
    face_image_gray = transform(face_image_gray).unsqueeze(0)
    with torch.no_grad():
        output = model(face_image_gray)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to convert image to bytes for downloading
def image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Display a simple header
st.markdown("<h1>Emotion Detection App</h1>", unsafe_allow_html=True)

# User instructions and description of the app
st.markdown("""
### How It Works:
1. **Upload an Image**: Upload a clear image that contains a human face. Supported file formats: **JPG, JPEG, PNG**. Maximum file size: **5 MB**.
2. **Detect Emotion**: Click the "Detect Emotion" button to process the image and classify the emotion.
3. **Emotion Results**: The detected emotion will be displayed with an emoji.
4. **Download Image**: Once processed, you can download the image with the detected face and emotion labeled.

If no face is detected, please ensure that the uploaded image contains a visible face.
""", unsafe_allow_html=True)


st.markdown("<h3>Upload Your Image</h3>", unsafe_allow_html=True)

# Image Upload Widget
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Define maximum file size (in bytes)
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

# Process the uploaded file
if uploaded_file is not None:
    try:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE:
            st.warning(f"File size exceeds {MAX_FILE_SIZE_MB} MB. Please upload a smaller image.")
        else:
            # Ensure the uploaded file is an image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Resize image for better processing
            if image_np.shape[1] > 600:
                image_np = cv2.resize(image_np, (600, int(image_np.shape[0] * (600 / image_np.shape[1]))))

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Add button to trigger emotion detection
            if st.button("Detect Emotion"):
                with st.spinner('Processing...'):
                    detected_image, faces = detect_face(image_np.copy())
                    st.image(detected_image, caption='Processed Image with Face Detection', use_container_width=True)

                # If faces are detected, classify emotions
                if faces:
                    for face in faces:
                        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                        face_image_rgb = Image.fromarray(image_np[y:y+h, x:x+w])
                        predicted_emotion = classify_emotion(face_image_rgb)

                        emotion_labels = {
                            0: ("Anger", "😡"),
                            1: ("Disgust", "🤢"),
                            2: ("Fear", "😱"),
                            3: ("Happiness", "😊"),
                            4: ("Sadness", "😢"),
                            5: ("Surprise", "😲"),
                            6: ("Neutral", "😐")
                        }
                        
                        predicted_emotion_label, emoji = emotion_labels[predicted_emotion]
                        st.markdown(f"<h2>{predicted_emotion_label} {emoji}</h2>", unsafe_allow_html=True)

                        # Provide emotion description
                        emotion_details = {
                            "Anger": "A strong feeling of displeasure or hostility.",
                            "Happiness": "A feeling of joy, contentment, or well-being.",
                            "Sadness": "A feeling of sorrow or unhappiness.",
                            "Fear": "A feeling of dread or anxiety.",
                            "Disgust": "A strong feeling of dislike or disapproval.",
                            "Surprise": "A feeling of astonishment or shock.",
                            "Neutral": "A state of being indifferent or without strong emotion."
                        }
                        st.write(f"**Emotion Details:** {emotion_details[predicted_emotion_label]}")

                        # Provide download option
                        download_image = image_to_bytes(Image.fromarray(detected_image))
                        st.download_button(label="Download Processed Image", data=download_image, file_name="processed_image.png", mime="image/png")
                else:
                    st.write("No face detected in the image. Please upload an image with a visible face.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
