#!/usr/bin/env python3

import random, os, numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ───── Config gerais ────────────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 96
BATCH_SIZE  = 64
EPOCHS      = 30
VAL_SPLIT   = 0.20
LR          = 1e-3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(SEED)

DATA_ROOT = Path("dataset")
TRAIN_DIR = DATA_ROOT / "training"
TEST_DIR  = DATA_ROOT / "test"


# ───── Dataset & transforms ─────────────────────────────────────────────
class LibrasDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ───── Coleta de caminhos (train/val) ───────────────────────────────────
class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
label2idx = {c:i for i,c in enumerate(class_names)}
idx2label = {i:c for c,i in label2idx.items()}

train_paths, train_labels = [], []
for cls in class_names:
    files = glob(str(TRAIN_DIR/cls/"*.jp*g")) + glob(str(TRAIN_DIR/cls/"*.png"))
    train_paths.extend(files)
    train_labels.extend([label2idx[cls]]*len(files))
combined = list(zip(train_paths, train_labels)); random.shuffle(combined)
train_paths, train_labels = map(list, zip(*combined))

# split
split_at = int(len(train_paths)*(1-VAL_SPLIT))
tr_p, vl_p = train_paths[:split_at], train_paths[split_at:]
tr_l, vl_l = train_labels[:split_at], train_labels[split_at:]

train_ds = LibrasDataset(tr_p, tr_l, train_tfms)
val_ds   = LibrasDataset(vl_p, vl_l, val_tfms)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train imgs: {len(train_ds)}  |  Val imgs: {len(val_ds)}")

# ───── Test set ─────────────────────────────────────────────────────────
test_paths, test_labels = [], []
for cls in class_names:
    files = glob(str(TEST_DIR/cls/"*.jp*g")) + glob(str(TEST_DIR/cls/"*.png"))
    test_paths.extend(files)
    test_labels.extend([label2idx[cls]]*len(files))
test_ds = LibrasDataset(test_paths, test_labels, val_tfms)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Test imgs: {len(test_ds)}")

# ───── Modelo simples ───────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        dummy = torch.zeros(1,3,IMG_SIZE,IMG_SIZE)
        flat  = self.features(dummy).view(1,-1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flat,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,n_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)


model = SimpleCNN(len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",0.3,3,1e-6,verbose=True)


# ───── Funções de treino/val ────────────────────────────────────────────
def run_epoch(model, loader, train=False):
    if train: model.train()
    else: model.eval()
    loss_sum, correct, total = 0,0,0
    all_preds, all_targs = [], []
    for x,y in (tqdm(loader,leave=False) if train else loader):
        x,y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out,y)
            if train:
                loss.backward(); optimizer.step()
        _, preds = out.max(1)
        loss_sum += loss.item()*x.size(0)
        correct  += (preds==y).sum().item()
        total    += x.size(0)
        all_preds.append(preds.cpu()); all_targs.append(y.cpu())
    acc = correct/total
    return loss_sum/total, acc, torch.cat(all_preds), torch.cat(all_targs)


best_val = 0.0
for epoch in range(1,EPOCHS+1):
    tr_loss,tr_acc,_,_ = run_epoch(model,train_dl,train=True)
    vl_loss,vl_acc,_,_ = run_epoch(model,val_dl,train=False)
    scheduler.step(vl_acc)
    print(f"[{epoch:02d}/{EPOCHS}]  "
          f"train loss:{tr_loss:.4f} acc:{tr_acc:.4%}  |  "
          f"val loss:{vl_loss:.4f} acc:{vl_acc:.4%}")
    if vl_acc>best_val:
        best_val=vl_acc
        torch.save(model.state_dict(),"best_libras_cnn.pt")
        print("  ↳ modelo salvo")


# ───── Avaliação no conjunto de teste ───────────────────────────────────
model.load_state_dict(torch.load("best_libras_cnn.pt"))
_,_, preds, targs = run_epoch(model,test_dl,train=False)


print("\n*** RESULTADOS NO TESTE ***")
print(classification_report(targs,preds,target_names=class_names,digits=4))


cm = confusion_matrix(targs,preds)
fig,ax = plt.subplots(figsize=(8,6)); im=ax.imshow(cm); plt.colorbar(im,ax)
ticks=np.arange(len(class_names))
ax.set_xticks(ticks); ax.set_xticklabels(class_names,rotation=90)
ax.set_yticks(ticks); ax.set_yticklabels(class_names)
plt.title("Matriz de Confusão (Teste)"); plt.xlabel("Predito"); plt.ylabel("Real")
plt.tight_layout(); plt.show()


# ───── Função de inferência individual ──────────────────────────────────
def predict(path):
    img = Image.open(path).convert("RGB")
    x   = val_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = torch.softmax(model(x),1)[0]
    idx  = out.argmax().item()
    return idx2label[idx], float(out[idx])

# exemplo rápido:
# letra, conf = predict("dataset/test/A/00123.jpg")
# print(letra, conf)
