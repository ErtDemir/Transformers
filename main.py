import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.flatten = nn.Flatten()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(128 * 128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = x.unsqueeze(1)  
        x = self.transformer(x)
        x = x.mean(dim=1)  
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


dataset = DeepfakeDataset(root_dir='C:\\Users\\WhirlErt\\Desktop\\Transformers\\DFDC', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


model = SimpleTransformer().to(device)


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


accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
