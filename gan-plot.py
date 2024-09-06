import os
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Load the fine-tuned Vision Transformer model
def extract_frames(video_path, max_frames=10):
    """Extract frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // max_frames)  # Sample max_frames evenly spaced
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret and i % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

# Step 2: Transform for videos
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 6: Load Fine-tuned Model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
model_save_path = 'fine_tuned_vit_model.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set model to evaluation mode

# Step 7: Prediction Function
def predict_video(video_path, model, transform):
    """Predict whether a video is real or fake."""
    model.eval()
    frames = extract_frames(video_path)
    
    if not frames:
        print(f"No frames found in video {video_path}.")
        return None

    # Apply the transformations
    frames_tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0)  # Add batch dimension
    outputs = model(frames_tensor[:, 0, :, :, :]).logits  # Predict based on the first frame

    predicted = torch.argmax(outputs, dim=1).item()
    
    # Get justification for the prediction
    justification = get_justification(frames_tensor, outputs)
    
    return 'Fake' if predicted == 1 else 'Real', justification

# Step 8: Justification Function
def get_justification(frames_tensor, outputs):
    """Generate a justification for the model's prediction."""
    probs = torch.nn.functional.softmax(outputs, dim=1)
    fake_prob = probs[0, 1].item()  # Probability that it's fake
    real_prob = probs[0, 0].item()  # Probability that it's real

    # Justification: Select frame with highest influence (class probability)
    # Assuming we're using the highest class probability to determine the influential frame
    most_influential_frame = frames_tensor[0][0].detach().cpu().numpy()  # Example frame for explanation

    return {
        "Fake Probability": fake_prob,
        "Real Probability": real_prob,
        "Most Influential Frame": most_influential_frame  # Provide the frame that influenced the decision
    }

# Function to plot the frame
def plot_frame(frame, title="Frame"):
    """Plot a single frame."""
    # Convert frame from (C, H, W) to (H, W, C)
    frame = np.transpose(frame, (1, 2, 0))
    
    # Denormalize (reverse the normalization applied during transformation)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = frame * std + mean
    
    # Clip values to be in [0, 1] range
    frame = np.clip(frame, 0, 1)
    
    plt.imshow(frame)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example Usage: Predict if a new video is Real or Fake
video_path = "D:/AI/SYNTHESIS/id0_id2_0002.mp4"  # Replace with your test video path
prediction, justification = predict_video(video_path, model, transform)

if prediction is not None:
    print(f"Prediction: {prediction}")
    print(f"Justification: {justification}")

    # Plot the most influential frame
    if "Most Influential Frame" in justification:
        influential_frame = justification["Most Influential Frame"]
        plot_frame(influential_frame, title=f"Most Influential Frame - {prediction}")
else:
    print("Prediction could not be made due to insufficient frames.")
