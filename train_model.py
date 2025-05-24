import random, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, DataCollatorWithPadding)
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ds = load_dataset("nvidia/HelpSteer2")
    train_data, val_data = ds["train"], ds["validation"]

    label_cols = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

    model_name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_pair(batch):
        return tok(batch["prompt"], batch["response"], truncation=True, max_length=512)

    train_data = train_data.map(tok_pair, batched=True, remove_columns=["prompt", "response"])
    val_data   = val_data.map(tok_pair, batched=True, remove_columns=["prompt", "response"])

    train_data.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "token_type_ids"] + label_cols,
                        output_all_columns=True)
    val_data.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "token_type_ids"] + label_cols,
                        output_all_columns=True)


    padder = DataCollatorWithPadding(tok, return_tensors="pt")

    def collate_fn(batch):
        # separate labels from features
        features = [{k: v for k, v in item.items() if k not in label_cols}
                    for item in batch]
        batch_padded = padder(features)
        for attr in label_cols:
            batch_padded[attr] = torch.tensor([item[attr] for item in batch],
                                            dtype=torch.long)
        return batch_padded

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    class MultiHeadReward(nn.Module):
        def __init__(self, enc_name):
            super().__init__()
            self.enc = AutoModel.from_pretrained(enc_name)
            h = self.enc.config.hidden_size
            

            self.layer_norm = nn.LayerNorm(h)
            self.dropout1 = nn.Dropout(0.3)  # Reduced first dropout
            self.dropout2 = nn.Dropout(0.5)  # Keep high dropout before final layer
            
            self.intermediate = nn.Linear(h, h // 2)
            self.heads = nn.ModuleList([nn.Linear(h // 2, 5) for _ in range(5)])
            
            for head in self.heads:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
            nn.init.xavier_uniform_(self.intermediate.weight)
            nn.init.zeros_(self.intermediate.bias)
            
        def forward(self, **enc_inputs):
            out = self.enc(**enc_inputs).last_hidden_state[:, 0]  # [CLS]
            out = self.layer_norm(out)
            out = self.dropout1(out)
            out = torch.relu(self.intermediate(out))
            out = self.dropout2(out)
            return [head(out) for head in self.heads]  # (B,5)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == "cpu":
        quit()
        
    def seconds_to_time(seconds):
        s = int(seconds) % 60
        m = int(seconds) // 60
        if m < 1:
            return f'{s}s'
        h = m // 60
        m = m % 60
        if h < 1:
            return f'{m}m{s}s'
        return f'{h}h{m}m{s}s'
        
    model  = MultiHeadReward(model_name).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss()

    epoch_train_loss = []
    epoch_val_loss= []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    epochs = 20
    max_batches = len(train_loader)
    start_time = time.time()
    
    for ep in range(1, epochs+1):
        print(f"Epoch: {ep}/{epochs}...", flush=True)
        model.train(); tr_loss = 0
        for i, batch in enumerate(train_loader):
            if len(train_loader) < max_batches:
                max_batches = len(train_loader)
            if i > max_batches:
                break
            
            inputs = {k: v.to(device) for k, v in batch.items()
                    if k in ["input_ids", "attention_mask", "token_type_ids"]}
            labels = {attr: batch[attr].to(device) for attr in label_cols}
            
            logits = model(**inputs)
            individual_losses = [criterion(l, labels[attr]) for l, attr in zip(logits, label_cols)]
            loss = sum(individual_losses) / 5.0
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            
            tr_loss += loss.item()
            elapsed = seconds_to_time(time.time() - start_time)
            if i % 100 == 0:
                print(f"  time: {elapsed} | batch: {i}/{max_batches}   ", end='\r', flush=True)
                
        avg_train_loss = tr_loss/max_batches
        epoch_train_loss.append(avg_train_loss)
        print(f"  train loss {avg_train_loss:.4f}                    ")

        # validation
        model.eval(); val_loss = 0
        print("  validating...", end='\r')
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()
                        if k in ["input_ids", "attention_mask", "token_type_ids"]}
                labels = {attr: batch[attr].to(device) for attr in label_cols}
                logits = model(**inputs)
                loss   = sum(criterion(l, labels[attr])
                            for l, attr in zip(logits, label_cols)) / 5.0
                val_loss += loss.item()
        avg_val_loss = val_loss/len(val_loader)
        epoch_val_loss.append(avg_val_loss)
        print(f"  val loss   {avg_val_loss:.4f}")
        
        old_lr = optim.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optim.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  Learning rate reduced to {new_lr:.2e}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  New best model saved! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {ep} epochs")
            break
        
        elapsed = seconds_to_time(time.time() - start_time)
        print(f"  time: {elapsed}", flush=True)
        
        with open('losses.txt', 'a') as f:
            f.write(f'{ep}, train: {epoch_train_loss[-1]}\n')
            f.write(f'{ep}, val: {epoch_val_loss[-1]}\n')
            
        if ep % 2 == 0:
            torch.save(model, f'models/metric_model_{ep}.pth')

    print("Done!", flush=True)
except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting gracefully.")