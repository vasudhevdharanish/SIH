import os
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn

# # Directory structure assumption
# real_videos_path = 'D:/AI/Celeb-real'
# fake_videos_path = 'D:/AI/SYNTHESIS'
# frame_extraction_dir = 'D:/AI/extract-frame'

# # Step 1: Dataset Preparation
# class DeepFakeDataset(Dataset):
#     def __init__(self, video_dir, label, transform=None):
#         self.video_dir = video_dir
#         self.label = label
#         self.videos = os.listdir(video_dir)
#         self.transform = transform

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx):
#         video_path = os.path.join(self.video_dir, self.videos[idx])
#         frames = self.extract_frames(video_path)
        
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]

#         return torch.stack(frames), self.label, video_path

#     def extract_frames(self, video_path, max_frames=10):
#         """Extract frames from a video."""
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         step = max(1, frame_count // max_frames)  # Sample max_frames evenly spaced
        
#         for i in range(frame_count):
#             ret, frame = cap.read()
#             if ret and i % step == 0:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                 frames.append(frame)
#             if len(frames) >= max_frames:
#                 break
#         cap.release()
#         return frames

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
# # Step 2: Transform for videos
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # Step 6: Load Fine-tuned Model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
model_save_path = 'fine_tuned_vit_model.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set model to evaluation mode

# Step 7: Prediction Function
def predict_video(video_path, model, transform):
    """Predict whether a video is real or fake."""
    model.eval()
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"No frames found in video {video_path}.")
        return None

    frames_tensor = torch.stack(frames).unsqueeze(0)  # Add batch dimension
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
    most_influential_frame = frames_tensor[0][0].detach().cpu().numpy()  # Example frame for explanation

    return {
        "Fake Probability": fake_prob,
        "Real Probability": real_prob,
        #"Most Influential Frame": most_influential_frame  # Provide the frame that influenced the decision
    }


# Example Usage: Predict if a new video is Real or Fake
video_path = "D:\AI\celeb-real\id2_0002.mp4"  # Replace with your test video path
prediction, justification = predict_video(video_path, model, transform)

if prediction is not None:
    print(f"Prediction: {prediction}")
    print(f"Justification: {justification}")
else:
    print("Prediction could not be made due to insufficient frames.")
