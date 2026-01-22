from data_loader.kt_loader import MyDataset, EmbeddingData, createDataLoader
from config import Config
from model.my_model import device, TextAKT3PL
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from loss.my_loss import SmoothBCE

def eval(model, test_loader, device):
    model.eval()
    # 确保指标在 GPU 上
    metrics = MetricCollection({
        'AUC': BinaryAUROC(),
        'ACC': BinaryAccuracy(threshold=0.5),
    }).to(device)
    
    with torch.no_grad():
        # 注意这里多接收了一个 eval_mask
        for c_emb, q_emb, responses, q_ids, eval_mask in test_loader:
            c_emb, q_emb = c_emb.to(device), q_emb.to(device)
            responses = responses.to(device)
            q_ids = q_ids.to(device)
            eval_mask = eval_mask.to(device)

            preds = model(c_emb, q_emb, responses) 

            # 核心逻辑：组合 Mask
            # 1. q_ids != -1 过滤掉 Padding
            # 2. eval_mask == 1 过滤掉重叠的历史部分
            valid_mask = (q_ids != -1) & (eval_mask == 1)

            # 使用组合后的 Mask 提取数据
            valid_preds = torch.masked_select(preds.squeeze(-1), valid_mask)
            valid_labels = torch.masked_select(responses.squeeze(-1), valid_mask)

            if valid_preds.numel() > 0:
                metrics.update(valid_preds, valid_labels.int())

    final = metrics.compute()
    auc, acc = final['AUC'].item(), final['ACC'].item()
    print(f"\n[Eval] ACC: {acc:.4f} | AUC: {auc:.4f}")
    model.train()
    return auc

def train(model, train_loader, optimizer, criterion, epochs, test_loader, scheduler=None):
    model.train()
    best_auc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0
        # 使用 tqdm 显示进度条
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (c_emb, q_emb, responses, q_ids, _) in pbar:
            c_emb, q_emb = c_emb.to(device), q_emb.to(device)
            responses, q_ids = responses.to(device), q_ids.to(device)
            
            optimizer.zero_grad()
            preds = model(c_emb, q_emb, responses)

            # Masking
            mask = (q_ids != -1).unsqueeze(-1)
            masked_preds = torch.masked_select(preds, mask)
            masked_labels = torch.masked_select(responses, mask)
            
            # 安全检查：防止 batch 全是 padding
            if masked_preds.numel() == 0:
                continue
                
            loss = criterion(masked_preds, masked_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 更新进度条显示的 Loss
            pbar.set_postfix({'loss': loss.item()})
        
        # Epoch 结束处理
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current LR: {current_lr:.6f}")

        # 评估 
        if (epoch + 1) % 15 == 0: 
            auc = eval(model, test_loader, device)
            if auc > best_auc:
                best_auc = auc
                # torch.save(model.state_dict(), "best_model.pth")
                print(f"*** New Best AUC: {best_auc:.4f} ***")


if __name__ == "__main__":
    name = Config.dataset.name[0]

    ed = EmbeddingData("NIPS2020", Config)
    train_loader, test_loader = createDataLoader("NIPS2020", Config, ed)
    
    model = TextAKT3PL(Config.model.inputDim, Config.model.modelDim, Config.model.baseScale, \
                       Config.model.blocks, Config.model.heads, Config.model.dropout,\
                       True, Config.dataset.historyLenth+1).to(device)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.training.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.training.epochs)
    criterion = SmoothBCE()
        
    train(model, train_loader, optimizer, criterion, Config.training.epochs, test_loader, scheduler)
