from utils.dataloader import create_dataloader
from face_yoromini import FaceYOLOMini, Detect
import torch, torch.optim as optim

def compute_loss(preds, targets):
    """
    YOLOv8論文のロジックに基づいた損失関数
    DFL+CIoU+BCE実装はここに記述
    """
    # TODO: YOLOv8論文ロジックに基づく実装
    return torch.tensor(0.0, requires_grad=True, device=preds.device)

def main():
    model = FaceYOLOMini().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    train_loader = create_dataloader('data/images/train', 'data/labels/train')
    
    for epoch in range(50):
        for imgs, targets in train_loader:  # imgs: (bs,3,640,640)
            preds, grid, stride = model(imgs.cuda())
            loss = compute_loss(preds, targets.cuda())  # DFL+CIoU+BCE
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f'runs/epoch{epoch}.pt')
        print(f"Epoch {epoch} completed. Model saved to runs/epoch{epoch}.pt")

if __name__ == "__main__":
    main()
