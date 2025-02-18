import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        for label, category in enumerate(['REAL', 'FAKE']):
            category_dir = os.path.join(root_dir, category, 'TRAIN')
            for video in os.listdir(category_dir):
                video_dir = os.path.join(category_dir, video)
                for frame in os.listdir(video_dir):
                    self.data.append((os.path.join(video_dir, frame), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformer Model with Xception + Transformer Encoder + Cross-Attention + FFN
class DeepfakeTransformer(nn.Module):
    def __init__(self):
        super(DeepfakeTransformer, self).__init__()
        
        # Xception-like feature extractor
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features
        self.feature_extractor[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1280, nhead=8), num_layers=2
        )
        
        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=1280, num_heads=8)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        batch_size, channels, height, width = features.size()

        # Flattening spatial dimensions
        features = features.view(batch_size, channels, -1).permute(2, 0, 1)

        # Transformer Encoder
        encoded_features = self.transformer_encoder(features)

        # Cross-Attention
        attn_output, _ = self.cross_attention(encoded_features, encoded_features, encoded_features)

        # Global average pooling
        pooled_output = attn_output.mean(dim=0)

        # Feed-Forward Network
        output = self.ffn(pooled_output)
        return output

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
dataset = DeepfakeDataset(root_dir='/DFDC', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize model
model = DeepfakeTransformer().to(device)

# Classification without training
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        print(predictions, labels)

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
