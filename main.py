import torch
import logging
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Model is on: {device}")

# Define transformations for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Dataset directory
dataset_dir = "dataset/"
if not os.path.exists(dataset_dir):
    logging.error(f"Dataset directory {dataset_dir} not found!")
    exit()

logging.info("Loading dataset...")
# Use ImageFolder to load data
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
logging.info(f"Dataset loaded with {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

# Split dataset into 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logging.info(f"Dataset split into {train_size} training and {val_size} validation samples.")

# DataLoader for batching
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
logging.info(f"Data loaders initialized with batch size {batch_size}.")

# Load EfficientNet model
logging.info("Loading EfficientNet model...")
model = EfficientNet.from_pretrained('efficientnet-b0')

# Modify classifier for binary classification
model._fc = nn.Linear(model._fc.in_features, 1)
model.to(device)
logging.info("Model modified for binary classification.")

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # More stable than BCELoss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
logging.info("Loss function and optimizer initialized.")

# Training loop
epochs = 10
logging.info(f"Starting training for {epochs} epochs...")
for epoch in range(epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    correct_preds, total_preds = 0, 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).float()  # Labels do not need unsqueeze here
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Model output
        loss = criterion(outputs, labels)  # BCEWithLogitsLoss expects output of shape [batch_size]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid before thresholding
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

        if batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}: Loss {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = correct_preds / total_preds
    logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss, val_correct_preds, val_total_preds = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct_preds += (preds == labels).sum().item()
            val_total_preds += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct_preds / val_total_preds
    logging.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the trained model
model_path = "drowsiness_detector_model.pth"
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved at {model_path}")
print(f"Model is on: {device}")
