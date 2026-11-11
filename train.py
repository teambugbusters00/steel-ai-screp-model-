"""Train a MobileNetV3 baseline on ImageFolder steel-scrap images."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    root=Path("data")
    if not (root/"train").exists() or not (root/"val").exists(): raise SystemExit("Add data/train/<class>/ and data/val/<class>/ image folders.")
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    train=datasets.ImageFolder(root/"train",transform=tfm); val=datasets.ImageFolder(root/"val",transform=tfm)
    if train.classes!=val.classes: raise SystemExit("Train and validation classes differ.")
    device="cuda" if torch.cuda.is_available() else "cpu"; model=models.mobilenet_v3_small(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(train.classes)); model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train(); total=0
        for x,y in DataLoader(train,batch_size=32,shuffle=True):
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); good=count=0
        with torch.no_grad():
            for x,y in DataLoader(val,batch_size=32):
                pred=model(x.to(device)).argmax(1).cpu(); good+=(pred==y).sum().item(); count+=y.numel()
        print(f"epoch {epoch+1}: train_loss={total/max(1,len(train)//32+1):.4f}, val_acc={good/max(1,count):.3f}")
    Path("artifacts").mkdir(exist_ok=True); torch.save({"state_dict":model.state_dict(),"classes":train.classes},"artifacts/model.pt")
if __name__=="__main__": main()