# Iteration 6: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 1. Load Data + Frozen Tokenizer (from Iteration 4)
# ============================================================
with h5py.File("../data/pong_frames.h5", "r") as f:
    raw_frames = f["frames"][:]

frames = raw_frames.astype(np.float32) / 255.0
frames = np.transpose(frames, (0, 3, 1, 2))
frames = torch.from_numpy(frames)

# --- Load frozen tokenizer (copy classes from Iter 4, load weights) ---
# tokenizer = SpaceTimeTokenizer().to(device)
# tokenizer.load_state_dict(torch.load("../iter4_spacetime_transformer/tokenizer.pt"))
# tokenizer.eval()
# for p in tokenizer.parameters(): p.requires_grad = False

VOCAB_SIZE = 1024
N_PATCHES = 64
NUM_ACTIONS = 3  # 0=stay, 1=up, 2=down

# ============================================================
# 2. Infer Pong Actions (heuristic)
# ============================================================
def find_paddle_y(frame_hw3):
    """Find the left paddle's vertical center in a (H, W, 3) frame."""
    # Left paddle is in the leftmost columns, bright pixels
    left_strip = frame_hw3[:, :8, :]  # leftmost 8 columns
    brightness = left_strip.mean(axis=-1)  # (H, 8)
    bright_rows = np.where(brightness.max(axis=1) > 0.5)[0]
    if len(bright_rows) == 0:
        return 32  # default to center
    return bright_rows.mean()

raw_frames_hwc = raw_frames.astype(np.float32) / 255.0  # (N, H, W, C)
print("Inferring actions from paddle movement...")
actions = []
FRAME_SKIP = 2
for i in range(0, len(raw_frames_hwc) - FRAME_SKIP, FRAME_SKIP):
    curr_y = find_paddle_y(raw_frames_hwc[i])
    next_y = find_paddle_y(raw_frames_hwc[i + FRAME_SKIP])
    dy = next_y - curr_y
    if dy < -1:
        actions.append(1)   # up
    elif dy > 1:
        actions.append(2)   # down
    else:
        actions.append(0)   # stay

actions = torch.tensor(actions, dtype=torch.long)
print(f"Actions: {len(actions)}, distribution: stay={( actions==0).sum()}, up={(actions==1).sum()}, down={(actions==2).sum()}")

# ============================================================
# 3. Tokenize Frames + Build Dataset
# ============================================================
NUM_FRAMES = 4

# Tokenize all clips (reuse approach from Iteration 5)
# all_tokens: (N, T, P)
# all_actions: (N, T-1) — one action per transition

# ... tokenize using frozen tokenizer as in Iter 5 ...
# For each clip starting at index i:
#   clip_tokens = tokenize frames[i, i+skip, i+2*skip, i+3*skip]
#   clip_actions = actions[i//skip, i//skip+1, i//skip+2]

class ActionTokenDataset(Dataset):
    def __init__(self, tokens, actions):
        """
        tokens: (N, T, P) — clip token indices
        actions: (N, T-1) — action for each frame transition
        """
        self.tokens = tokens
        self.actions = actions

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        clip = self.tokens[idx]        # (T, P)
        acts = self.actions[idx]       # (T-1,)
        context = clip[:-1]            # (T-1, P)
        target = clip[1:]             # (T-1, P)
        return context, target, acts

# token_dataset = ActionTokenDataset(all_tokens, all_actions)
# token_loader = DataLoader(token_dataset, batch_size=64, shuffle=True)

# ============================================================
# 4. FiLM Module
# ============================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        """
        x: (B, seq_len, d_model)
        cond: (B, d_model) — conditioning signal
        """
        gamma = 1 + self.scale(cond).unsqueeze(1)  # (B, 1, d_model), centered at 1
        beta = self.shift(cond).unsqueeze(1)        # (B, 1, d_model)
        return gamma * x + beta

# ============================================================
# 5. Space-Time Block with FiLM
# ============================================================
class SpaceTimeBlockFiLM(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.film = FiLM(d_model, d_model)

    def forward(self, x, T, P, action_embed, causal_mask=None):
        B_orig, _, E = x.shape

        # Spatial attention
        x_s = x.reshape(B_orig * T, P, E)
        ln_x = self.ln1(x_s)
        x_s = x_s + self.spatial_attn(ln_x, ln_x, ln_x)[0]
        x = x_s.reshape(B_orig, T * P, E)

        # Temporal attention (causal)
        x_t = x.reshape(B_orig, T, P, E).permute(0, 2, 1, 3).reshape(B_orig * P, T, E)
        ln_x = self.ln2(x_t)
        x_t = x_t + self.temporal_attn(ln_x, ln_x, ln_x, attn_mask=causal_mask)[0]
        x = x_t.reshape(B_orig, P, T, E).permute(0, 2, 1, 3).reshape(B_orig, T * P, E)

        # FFN + FiLM
        ffn_out = self.ffn(self.ln3(x))
        ffn_out = self.film(ffn_out, action_embed)  # ← action modulation
        x = x + ffn_out
        return x

# ============================================================
# 6. Dynamics Model with Action Conditioning
# ============================================================
class ActionDynamicsModel(nn.Module):
    def __init__(self, vocab_size=1024, num_actions=3, d_model=128,
                 n_heads=4, n_layers=4, n_patches=64, max_frames=4):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.action_embed = nn.Embedding(num_actions, d_model)

        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_patches, d_model) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            SpaceTimeBlockFiLM(d_model, n_heads) for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_frames, max_frames), diagonal=1).bool()
        )

    def forward(self, tokens, actions):
        """
        tokens: (B, T, P) integer indices
        actions: (B, T) integer action IDs
        → logits: (B, T, P, vocab_size)
        """
        B, T, P = tokens.shape

        x = self.token_embed(tokens)  # (B, T, P, d_model)
        x = x + self.spatial_pos[:, :, :P] + self.temporal_pos[:, :T]
        x = x.reshape(B, T * P, self.d_model)

        # Average action embedding across timesteps for conditioning
        act_emb = self.action_embed(actions).mean(dim=1)  # (B, d_model)

        causal = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, T, P, act_emb, causal_mask=causal)

        x = x.reshape(B, T, P, self.d_model)
        return self.head(self.ln_final(x))

dynamics = ActionDynamicsModel().to(device)
print(f"Dynamics params: {sum(p.numel() for p in dynamics.parameters()):,}")

# ============================================================
# 7. Train
# ============================================================
optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-4)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for context, target, acts in token_loader:
        context = context.to(device)
        target = target.to(device)
        acts = acts.to(device)

        logits = dynamics(context, acts)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.4f}")

# ============================================================
# 8. Interactive Generation — Different Actions, Different Outcomes
# ============================================================
dynamics.eval()

def generate_with_actions(seed_tokens, action_sequence):
    """Generate frames following a sequence of actions."""
    tokens = seed_tokens.clone()
    for action_id in action_sequence:
        T_curr = tokens.shape[1]
        # Create action tensor: all same action for simplicity
        acts = torch.full((1, T_curr), action_id, device=device)

        logits = dynamics(tokens, acts)
        next_tokens = logits[:, -1, :, :].argmax(dim=-1).unsqueeze(1)
        tokens = torch.cat([tokens, next_tokens], dim=1)
    return tokens

with torch.no_grad():
    # Encode seed
    seed_clip = frames[:3 * 2:2].unsqueeze(0).to(device)
    _, seed_tokens = tokenizer.encode(seed_clip)

    # Generate with three different action plans
    plans = {
        "UP × 4":   [1, 1, 1, 1],
        "DOWN × 4": [2, 2, 2, 2],
        "STAY × 4": [0, 0, 0, 0],
    }

    fig, axes = plt.subplots(3, 7, figsize=(21, 9))

    for row, (label, plan) in enumerate(plans.items()):
        gen_tokens = generate_with_actions(seed_tokens, plan)
        gen_pixels = tokens_to_pixels(gen_tokens)  # reuse from Iter 5

        for t in range(gen_pixels.shape[1]):
            img = gen_pixels[0, t].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, t].imshow(img)
            axes[row, t].axis("off")
            if t < 3:
                axes[row, t].set_title(f"Seed {t}", fontsize=9)
            else:
                axes[row, t].set_title(f"Gen (act={plan[t-3]})", fontsize=9)

        axes[row, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=60)

    plt.suptitle("Same seed, different actions → different futures", fontsize=14)
    plt.tight_layout()
    plt.savefig("action_conditioning.png", dpi=150)
    plt.show()
```

## What You Should Observe

1. **Different actions produce different futures** — pressing UP moves the paddle up, DOWN moves it down, STAY keeps it stationary. The model has learned to associate actions with outcomes.

2. **FiLM is elegant** — just ~2 small linear layers per block, but the conditioning is strong. The model generates clearly different scenes for different actions.

3. **We cheated** — we used a heuristic to extract action labels from the video. For real internet video (YouTube gameplay, movies), you don't have this. Someone playing Mario on YouTube doesn't tell you which buttons they pressed.

## Why This Matters

You now have a controllable world model. But it depends on having action labels, which limits you to games where you can record inputs. The Genie paper's key insight is:

> *If you look at two consecutive frames, you can figure out what "action" must have happened between them — even without ever seeing the actual button press.*

Frame_t shows the paddle at y=30. Frame_t+1 shows it at y=25. Something moved it up — call that "action 1." You don't need to know the player pressed UP; you just observe the effect.

**Next iteration**: build a Latent Action Model that learns to infer discrete actions from pairs of frames, completely unsupervised.
