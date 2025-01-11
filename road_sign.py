import os
import random
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import xml.etree.ElementTree as ET

# Paths for annotations and images
anno_path = Path(r"D:\CV_PROJ\road\annotations")
images_path = Path(r"D:\CV_PROJ\road\images")

# Function to get file list
def filelist(root: Path, file_type: str) -> list:
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

# Function to generate DataFrame from XML annotations
def generate_train_df(anno_path: Path) -> pd.DataFrame:
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_file in annotations:
        root = ET.parse(anno_file).getroot()
        anno = {
            'filename': Path(images_path / root.find("./filename").text),
            'class': root.find("./object/name").text,
        }
        anno_list.append(anno)
    return pd.DataFrame(anno_list)

# Generate DataFrame from annotations
df_train = generate_train_df(anno_path)

# Convert class names to integer labels
class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

# Define the custom dataset
class RoadSignDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))  # Resize the image

        label = self.df.iloc[idx]['class']
        image = image.astype(np.float32) / 255.0  # Normalize the image

        # Convert to tensor
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # Change shape to (C, H, W)
        label_tensor = torch.tensor(label)  # Convert label to tensor

        return image_tensor, label_tensor  # Return image and label tensors

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 37 * 37, 256)  # Adjust input size based on the output size after convolutions
        self.fc2 = nn.Linear(256, len(class_dict))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 37 * 37)  # Flatten the output
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)
        return x

# Create DataLoader
batch_size = 32
train_dataset = RoadSignDataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Function to predict class from an image
def predict_class(image_path: str):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))  # Resize the image
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Test the prediction function
test_image_path = r"D:\CV_PROJ\road\images\road64.png"  # Replace with your image path
predicted_class = predict_class(test_image_path)
class_names = {v: k for k, v in class_dict.items()}
print(f'The predicted class is: {class_names[predicted_class]}')
# Saving the model as .pth
torch.save(model.state_dict(), "simple_cnn1_model.pth")

