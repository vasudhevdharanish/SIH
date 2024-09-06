import os
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn

# Directory structure assumption
real_videos_path = 'D:/AI/Celeb-real'
fake_videos_path = 'D:\AI\SYNTHESIS'
frame_extraction_dir = 'D:/AI/extract-frame'

# Step 1: Dataset Preparation
class DeepFakeDataset(Dataset):
    def __init__(self, video_dir, label, transform=None):
        self.video_dir = video_dir
        self.label = label
        self.videos = os.listdir(video_dir)
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.videos[idx])
        frames = self.extract_frames(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return torch.stack(frames), self.label, video_path

    def extract_frames(self, video_path, max_frames=10):
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

# Step 2: Transform and DataLoader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

real_dataset = DeepFakeDataset(video_dir=real_videos_path, label=0, transform=transform)
fake_dataset = DeepFakeDataset(video_dir=fake_videos_path, label=1, transform=transform)

# Combine datasets and create data loaders
dataset = real_dataset + fake_dataset
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 3: Vision Transformer Model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2,ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Use pre-trained Vision Transformer model and fine-tune it
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Step 4: Training loop
model.train()
for epoch in range(5):  # Train for 5 epochs
    running_loss = 0.0
    for frames, labels, _ in train_loader:
        # Assume we're classifying the first frame (for simplicity)
        outputs = model(frames[:, 0, :, :, :]).logits
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader)}")

# Step 5: Save the fine-tuned model
model_save_path = 'fine_tuned_vit_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
