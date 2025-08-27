import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === FocalLoss 정의 ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * CE_loss
        else:
            loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# === 설정 ===
num_epochs = 10
batch_size = 32
num_classes = 4
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CUDA 상태 확인 ===
print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] CUDA 사용 불가 → CPU 사용")
    
print(f"[INFO] 디바이스 사용 중: {device}")

train_dir = "./dataset/train"
val_dir = "./dataset/val"
save_path = "./dataset/models/model_mobilenetv2.pth"

# === 데이터 전처리 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# === 데이터 로딩 ===
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"[INFO] 클래스 목록: {train_dataset.classes}")
print(f"[INFO] 클래스 인덱스 매핑: {train_dataset.class_to_idx}")
class_names = train_dataset.classes

# === alpha 가중치 정의 (클래스별) ===
alpha_map = {
    "obstacle": 0.4,
    "red": 0.2,
    "yellow": 0.2,
    "green": 0.2
}
alpha = [alpha_map[class_name] for class_name in class_names]
print(f"[INFO] 적용된 FocalLoss alpha 값: {alpha}")

# === MobileNetV2 모델 로드 및 수정 ===
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# === 손실 함수 및 옵티마이저 설정 ===
criterion = FocalLoss(alpha=alpha, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=lr)

# === 학습 루프 ===
for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}] 시작 ===============================")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # === 검증 루프 ===
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# === 모델 저장 ===
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"\n[완료] 모델 저장됨 → {save_path}")

