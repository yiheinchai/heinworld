import h5py
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/pong_frames.h5"
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
LOG_EVERY = 50  # steps between progress updates

# ── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ── Dataset ─────────────────────────────────────────────────────────────────
class PongDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, index):
        return self.data[index], self.data[index + 1]


# ── Model ───────────────────────────────────────────────────────────────────
class NextFrameMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(12288, 256)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(256, 12288)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = x.view(batch_size, 3, 64, 64)
        return x


# ── Helpers ─────────────────────────────────────────────────────────────────
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def progress_bar(current, total, width=30):
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {current}/{total}"


def print_header():
    cols = os.get_terminal_size(fallback=(80, 24)).columns
    print("=" * cols)
    print("  Next-Frame Prediction Training".center(cols))
    print("=" * cols)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR:         {LR}")
    print("=" * cols)
    print()


def print_epoch_header(epoch):
    print(f"  Epoch {epoch}/{NUM_EPOCHS}")
    print(f"  {'─' * 60}")


def print_step_line(step, total_steps, loss, avg_loss, elapsed, steps_per_sec):
    bar = progress_bar(step, total_steps, width=25)
    eta = format_time((total_steps - step) / steps_per_sec) if steps_per_sec > 0 else "?"
    line = (
        f"\r  {bar}  "
        f"loss: {loss:.5f}  "
        f"avg: {avg_loss:.5f}  "
        f"[{format_time(elapsed)} < {eta}, {steps_per_sec:.1f} it/s]"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def print_epoch_summary(epoch, avg_loss, best_loss, elapsed):
    improved = " *" if avg_loss <= best_loss else ""
    print(f"\n  Epoch {epoch} done  |  avg loss: {avg_loss:.6f}{improved}  |  time: {format_time(elapsed)}")
    print()


def print_final_summary(epoch_losses, total_time):
    cols = os.get_terminal_size(fallback=(80, 24)).columns
    print("=" * cols)
    print("  Training Complete".center(cols))
    print("=" * cols)
    print(f"  Total time:  {format_time(total_time)}")
    print(f"  Best loss:   {min(epoch_losses):.6f} (epoch {np.argmin(epoch_losses) + 1})")
    print(f"  Final loss:  {epoch_losses[-1]:.6f}")
    print()

    # Mini loss curve in terminal
    print("  Loss curve:")
    chart_height = 8
    chart_width = min(len(epoch_losses), 50)
    vals = epoch_losses
    lo, hi = min(vals), max(vals)
    spread = hi - lo if hi != lo else 1.0

    for row in range(chart_height, -1, -1):
        threshold = lo + (row / chart_height) * spread
        if row == chart_height:
            label = f"  {hi:.5f} │"
        elif row == 0:
            label = f"  {lo:.5f} │"
        else:
            label = f"          │"
        chars = ""
        for v in vals:
            chars += "█" if v >= threshold else " "
        print(f"{label}{chars}")

    print(f"          └{'─' * len(vals)}")
    epoch_labels = "".join(str((i + 1) % 10) for i in range(len(vals)))
    print(f"           {epoch_labels}")
    print(f"           {'(epoch)':^{len(vals)}}")
    print()
    print("=" * cols)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print_header()

    # Load data
    print("  Loading data...", end="", flush=True)
    t0 = time.time()
    with h5py.File(DATA_PATH, "r") as f:
        frames = f["frames"][:]
    frames_trch = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    frames_trch = frames_trch.to(device)
    print(f" {len(frames_trch)} frames loaded in {time.time() - t0:.1f}s")
    print(f"  Shape: {tuple(frames_trch.shape)}")
    print()

    dataset = PongDataset(frames_trch)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    steps_per_epoch = len(loader)

    model = NextFrameMLP().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")
    print()

    # Training loop
    epoch_losses = []
    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print_epoch_header(epoch)
        batch_losses = []
        epoch_start = time.time()

        for step_in_epoch, (batch_input, batch_target) in enumerate(loader, 1):
            pred = model(batch_input)
            loss = loss_fn(pred, batch_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            if step_in_epoch % LOG_EVERY == 0 or step_in_epoch == steps_per_epoch:
                elapsed = time.time() - epoch_start
                avg = np.mean(batch_losses)
                speed = step_in_epoch / elapsed if elapsed > 0 else 0
                print_step_line(step_in_epoch, steps_per_epoch, loss.item(), avg, elapsed, speed)

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
        print_epoch_summary(epoch, avg_loss, best_loss, time.time() - epoch_start)

    print_final_summary(epoch_losses, time.time() - train_start)


if __name__ == "__main__":
    main()
