# Added better data augmentation
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root_dir = '/home/guest/Desktop/DENEME/DFDC'
 
# === Dataset Definition ===
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
 
 
# === Model Definition ===
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Boyutu küçültüp sabitleme
        self.flatten = nn.Flatten()
       
        self.linear_proj = nn.Linear(8 * 8 * 128, 128)  # Embedding'e uygun hale getirme
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(128, 2)  # Çıkış katmanı
 
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear_proj(x).unsqueeze(1)  # Transformer için uygun hale getir
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
 
 
 
# === Hyperparameters ===
batch_size = 64
learning_rate = 0.0005  # Küçültüldü!
num_epochs = 1000
 
# === Data Augmentation & Normalization ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
 
 
# === Dataset & DataLoader ===
dataset = DeepfakeDataset(root_dir= dataset_root_dir, transform=transform)
train_size = int(0.6 * len(dataset))  
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# === Model, Loss & Optimizer ===
model = TransformerModel().to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  
 
# === Training Loop ===
best_val_acc = 0.0  
train_accuracies = []
val_accuracies = []
 
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct_train = 0
    total_train = 0
 
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
   
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
 
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)
 
        train_pbar.set_postfix(loss=running_loss / total_train, acc=100 * correct_train / total_train)
 
 
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100* correct_train / total_train
    train_accuracies.append(epoch_accuracy)
 
    # === Validation Loop ===
    model.eval()
    correct_val = 0
    total_val = 0
 
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation", unit="batch")
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)
 
    val_acc = 100 * correct_val / total_val
    val_accuracies.append(val_acc)
    log = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss:.4f} | Train Acc: {100 * correct_train / total_train:.2f}% | Val Acc: {val_acc:.2f}% \n"
    print(log)
    with open(f'training_log_{num_epochs}.txt', 'a') as f:
        f.write(log)
 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_v2.pth")
        print("Best model saved!")
   
# === Test Loop ===
model.load_state_dict(torch.load("best_model_v2.pth"))
model.eval()
correct_test = 0
total_test = 0
 
with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    for images, labels in test_pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test += labels.size(0)
   
    test_acc = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_acc:.2f}%")
   
plt.figure()
plt.plot(range(1,num_epochs + 1), train_accuracies, marker="o", linestyle="-", label="Training Accuracy")
plt.plot(range(1,num_epochs + 1), val_accuracies, marker="x", linestyle="--", label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.savefig(f'training_validation_accuracy_plot_{num_epochs}_epoch.png')
plt.close
 
 
print("Training Finished!")