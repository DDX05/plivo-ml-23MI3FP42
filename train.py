import torch
import math
from tokenizer import Tokenizer
from model import GPT

def get_lr(step, max_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    device = 'cpu'
    torch.manual_seed(1337)
    
    # 1. Load Data
    with open('data/train_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Setup Tokenizer (Vocab 1024 to save params for depth while compressing Hindi bytes)
    tokenizer = Tokenizer(vocab_size=1024)
    print("Training BPE Tokenizer...")
    tokenizer.train(text)
    
    print("Encoding corpus...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Corpus size: {len(data)} tokens")

    # 3. Setup Model Config
    # This guarantees < 2,000,000 params (Specifically ~1,967,808)
    config = {
        'vocab_size': tokenizer.vocab_size,
        'n_embd': 192,
        'n_layer': 4,
        'n_heads': 6,
        'n_inner': 512, # SwiGLU hidden dim
        'dropout': 0.1,
        'block_size': 256
    }
    
    model = GPT(config).to(device)
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params}")
    assert num_params <= 2000000, "Parameter limit exceeded!"

    # 4. Training Hyperparameters
    max_steps = 2000
    batch_size = 16
    grad_accum_steps = 4 # Effective batch size = 64
    max_lr = 3e-3
    min_lr = 1e-4
    warmup_steps = 200
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    def get_batch():
        ix = torch.randint(len(data) - config['block_size'], (batch_size,))
        x = torch.stack([data[i:i+config['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+config['block_size']+1] for i in ix])
        return x.to(device), y.to(device)

    # 5. Training Loop
    model.train()
    for step in range(max_steps):
        lr = get_lr(step, max_steps, warmup_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for _ in range(grad_accum_steps):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == max_steps - 1:
            print(f"Step {step:4d} | Loss: {loss_accum:.4f} | LR: {lr:.6f}")

    # 6. Save Checkpoint (Including Tokenizer merges and Step Count)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': max_steps,
        'tokenizer_merges': tokenizer.merges 
    }
    torch.save(checkpoint, 'ckpt.pt')
    print("Training complete. Checkpoint saved to ckpt.pt")

if __name__ == '__main__':
    main()