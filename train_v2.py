import torch
from torch.nn import functional as F
from model_v2 import TinyTransformer

torch.manual_seed(42)

# Small original training dataset for experimentation
text = (
    "hello world. i am making my own ai. "
    "my model learns patterns from text. "
    "this is a tiny transformer trained from scratch. "
    "learning takes time and practice. "
) * 300

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

model = TinyTransformer(len(chars))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
block_size = model.block_size
batch_size = 16

print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Training from random initialization...")

model.train()
for step in range(1500):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])

    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 150 == 0:
        print(f"Step {step+1}/1500 | Loss: {loss.item():.4f}")

torch.save(
    {"model": model.state_dict(), "stoi": stoi, "itos": itos},
    "/content/transformer_v2.pt"
)
print("Training finished! Saved transformer_v2.pt")
