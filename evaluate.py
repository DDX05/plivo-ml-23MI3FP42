import argparse
import math
import torch
from model import GPT
from tokenizer import Tokenizer

def evaluate_bpb(model, tokenizer, text_file, device='cpu'):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Calculate raw byte length for BPB denominator
    num_bytes = len(text.encode('utf-8'))
    
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long).to(device)
    
    block_size = model.config['block_size']
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        # Evaluate in non-overlapping chunks
        for i in range(0, len(data) - 1, block_size):
            end_idx = min(i + block_size, len(data) - 1)
            if end_idx - i <= 0:
                break
                
            x = data[i:end_idx].unsqueeze(0)
            y = data[i+1:end_idx+1].unsqueeze(0)
            
            logits, loss = model(x, y)
            
            chunk_tokens = y.numel()
            total_loss += loss.item() * chunk_tokens
            total_tokens += chunk_tokens
            
    # Cross entropy calculates natural log (base e).
    # Convert NLL to base 2 for bits.
    total_bits = (total_loss) / math.log(2)
    bpb = total_bits / num_bytes
    
    return bpb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Accept both -checkpoint (requested in PDF) and --checkpoint (standard)
    parser.add_argument('-checkpoint', '--checkpoint', type=str, default='ckpt.pt', help='Path to checkpoint file')
    parser.add_argument('--text_file', type=str, required=True, help='Path to text file for evaluation')
    args = parser.parse_args()

    device = 'cpu'

    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct Model
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Reconstruct Tokenizer (using embedded merges from checkpoint)
    tokenizer = Tokenizer(vocab_size=checkpoint['config']['vocab_size'])
    if 'tokenizer_merges' in checkpoint:
        tokenizer.load(merges_dict=checkpoint['tokenizer_merges'])
    
    print(f"Evaluating on {args.text_file}...")
    bpb = evaluate_bpb(model, tokenizer, args.text_file, device)
    
    print(f"Final Bits Per Byte (BPB): {bpb:.4f}")