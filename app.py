import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path

# Define the class dictionary
class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
class_names = {v: k for k, v in class_dict.items()}

# Load your model (you'll need to train and save it first)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * 37 * 37, 256)
        self.fc2 = torch.nn.Linear(256, len(class_dict))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.nn.ReLU()(self.conv1(x)))
        x = self.pool(torch.nn.ReLU()(self.conv2(x)))
        x = self.pool(torch.nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 37 * 37)  # Flatten the output
        x = self.dropout(torch.nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the pre-trained model (make sure to adjust the path)
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Function to predict class from an image
def predict_class(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize the image
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Streamlit app
st.title("Road Sign Classification")

st.write("Upload an image of a road sign and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(opencv_image, channels="BGR", caption="Uploaded Image.", use_column_width=True)

    # Predict the class
    predicted_class = predict_class(opencv_image)

    # Show the prediction
    st.write(f"Predicted class: **{predicted_class}**")
