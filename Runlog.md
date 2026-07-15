Training Run Log

Run 1: Baseline Execution

Hypothesis: The baseline uses a vanilla GPT with absolute embeddings, a zero learning-rate schedule, and a pure byte-level tokeniser (vocab=256). Since Devanagari Hindi text uses 3 bytes per character, a byte-level tokenizer brutally limits the context window and forces the model to learn raw UTF-8 construction rather than semantics.

What Changed: Ran the provided starter script to establish a baseline.

Dev BPB Before/After: N/A -> ~2.15 bpb.

Conclusion: The baseline is heavily constrained by sequence lengths and inefficient parameter allocations.

Run 2: Implementation of BPE Tokenizer

Hypothesis: Implementing a lightweight Byte-Pair Encoding (BPE) tokenizer will allow the model to group frequent byte pairs (especially Devanagari UTF-8 triplets) into single tokens. This artificially expands the model's receptive field within the 256-token context block without altering the max_steps cap.

What Changed: Replaced the dummy tokenizer with a custom BPE Tokenizer built purely in the Python standard library. Set vocabulary to 1024 (256 base bytes + 768 merges).

Dev BPB Before/After: 2.15 -> 1.82 bpb.

Conclusion: Massive improvement. BPE effectively compressed the sequence lengths, yielding a denser semantic representation for the model.

Run 3: Weight Tying

Hypothesis: The 2M parameter cap is the ultimate bottleneck. The embedding matrix and the final LM head map to the same dimensions. By tying their weights (lm_head.weight = token_emb.weight), we save 1024 x 192 = 196K parameters (approx).

What Changed: Implemented Weight Tying. Re-allocated the saved parameter budget to increase the number of layers from 3 to 4.Total params: 1,967,808.

Dev BPB Before/After: 1.82 -> 1.55 bpb.

Conclusion: Free parameter budget allows a deeper network, dropping the loss significantly.

Run 4: Modernization - RoPE & RMSNorm

Hypothesis: Standard positional embeddings take time to learn, and we only have 2,000 steps. Replacing absolute positions with Rotary Positional Embeddings (RoPE) injects structural priors that converge faster. Switching to RMSNorm improves compute efficiency on the CPU.

What Changed: Added RMSNorm and RoPE to the attention mechanism.

Dev BPB Before/After: 1.55 -> 1.41 bpb.

Conclusion: RoPE allows the model to generalise sequence positions much faster within the 2000-step limit compared to learning absolute embeddings from scratch.

Run 5: Modernization - SwiGLU Activations

Hypothesis: SwiGLU activations (used in LLaMA) provide better gradient flow and higher parameter density/efficiency than standard GeLU or ReLU.

What Changed: Replaced standard MLPs with a SwiGLU layer (hidden dim 512).

Dev BPB Before/After: 1.41 -> 1.33 bpb.

Conclusion: Consistent improvement. The LLaMA-style block is far superior to standard GPT-2 blocks for micro-models.

Run 6: Optimization & LR Scheduling

Hypothesis: 2000 steps is extremely short. The baseline optimizer likely gets stuck or diverges. A Cosine Annealing schedule with a linear warmup prevents early divergence and smoothly decays to find a local minimum. Additionally, simulating a larger batch size reduces gradient noise.

What Changed: Swapped to AdamW (weight_decay=0.1). Implemented Gradient Accumulation (grad_accum_steps=4, effectively batch size 64). Added a Cosine decay LR schedule peaking at 3e-3 after a 200-step warmup.

Dev BPB Before/After: 1.33 -> 1.05 bpb.

Conclusion: The LR scheduler and batch-size simulation provided the final massive leap in performance. The model reaches a much tighter, more stable convergence within the exact 2000-step budget.