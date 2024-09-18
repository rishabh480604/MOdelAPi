import torch
# import torch # Import the torch module
import torch.nn as nn # Import the nn module
import cv2
# import torch
import numpy as np


def load_facial_expression_model(model_path="fer2013_model_complete.pth"):
    """
    Loads the pre-trained facial expression recognition model from the specified path.

    Args:
        model_path (str, optional): Path to the saved model state dictionary (default: "fer2013_model.pth").

    Returns:
        torch.nn.Module: The loaded facial expression recognition model.
    """
    facial_expression_model = FacialExpressionModel()
    facial_expression_model.load_state_dict(torch.load(model_path))
    return facial_expression_model



# ... rest of your code

# Define the CNN model for facial expression recognition
class FacialExpressionModel(nn.Module): # Define the FacialExpressionModel class
    def __init__(self):
        super(FacialExpressionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)

        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)

        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)

        x = x.view(-1, 256 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Initialize the facial expression recognition model
facial_expression_model = FacialExpressionModel()



# Define emotion labels (modify these based on your model's output)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the pre-trained emotion detection model
model = load_facial_expression_model("fer2013_model.pth")  # Replace with your model path
model.eval() # Set the model to evaluation mode

# Function to detect emotion from an image
def detect_emotion(image_path):
  # Read the image in grayscale (may vary depending on model input)
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  # Preprocess the image (resize, normalize) based on your model's requirements
  img = cv2.resize(img, (48, 48))  # Example resizing (adjust as needed)
  img = img.astype('float32') / 255.0
  img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Reshape for model input (adjust as needed)

  # Make prediction using the model
  with torch.no_grad():
    prediction = model(img)

  # Get the predicted emotion label with the highest probability
  emotion_index = torch.argmax(prediction).item()
  emotion = emotions[emotion_index]

  return emotion

"""
# Example usage: Replace 'path/to/image.jpg' with your image path
image_path = '/content/4002.jpeg'
predicted_emotion = detect_emotion(image_path)

print("Predicted emotion:", predicted_emotion)
"""
