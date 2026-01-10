"""
train_crnn.py
Plantilla PyTorch para clasificador simple de números 1..99.
Requiere un CSV con `image_path,label`.
"""
import argparse, os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class NumDataset(Dataset):
    def __init__(self,csv_path, transform=None):
        import csv
        self.items=[]
        with open(csv_path,newline='') as f:
            r=csv.reader(f)
            for row in r:
                if not row: continue
                self.items.append((row[0], int(row[1])))
        self.transform=transform
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        p,label=self.items[idx]
        img=Image.open(p).convert('RGB')
        if self.transform: img=self.transform(img)
        return img, label

def build_model(num_classes=100):
    m=models.mobilenet_v2(pretrained=True)
    m.classifier[1]=nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def train(args):
    tr=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    train_ds=NumDataset(args.train_csv, transform=tr)
    val_ds=NumDataset(args.val_csv, transform=tr)
    td=DataLoader(train_ds, batch_size=32, shuffle=True)
    vd=DataLoader(val_ds, batch_size=32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(100).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    for e in range(args.epochs):
        model.train()
        for xb,yb in td:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); out=model(xb); loss=loss_fn(out,yb); loss.backward(); opt.step()
        print(f'Epoch {e} done')
    torch.save(model.state_dict(), args.out)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--out', required=True)
    args=p.parse_args()
    train(args)
