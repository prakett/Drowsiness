import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Dataset ===
dataset = ImageFolder(root="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# === Load Trained Model ===
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load("drowsiness_detector_model.pth", map_location=device))
model.to(device)
model.eval()

# === Prediction ===
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Metrics ===
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Open_Eyes", "Closed_Eyes"])

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
