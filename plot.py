import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("checkpoints/training_log.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ── Top: loss curves ──────────────────────────────────────────────────────────
ax1.plot(df["step"], df["train_total"], label="Train Total", color="blue", linewidth=2)
ax1.plot(
    df["step"],
    df["train_recon"],
    label="Train Recon",
    color="skyblue",
    linewidth=1.5,
    linestyle="--",
)
ax1.plot(
    df["step"],
    df["train_delta"],
    label="Train Delta",
    color="purple",
    linewidth=1.5,
    linestyle="--",
)
ax1.plot(df["step"], df["val_loss"], label="Val Loss", color="orange", linewidth=1)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_title("Training Progress", fontsize=14)

# ── Bottom: learning rate ─────────────────────────────────────────────────────
ax2.plot(df["step"], df["learning_rate"], color="green", linewidth=1.5)
ax2.set_title("Learning Rate", fontsize=14)
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Learning Rate", color="green", fontsize=12)
ax2.tick_params(axis="y", labelcolor="green")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
