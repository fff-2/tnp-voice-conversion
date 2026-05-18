import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/training_log.csv")

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df["step"], df["train_loss"], label="Train Loss (Masked L1)", color="blue", linewidth=2)
ax1.plot(df["step"], df["val_loss"], label="Val Loss", color="orange", linewidth=2)
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Masked L1 Loss", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(df["step"], df["learning_rate"], label="Learning Rate", color="green", linestyle="--", linewidth=1.5)
ax2.set_ylabel("Learning Rate", color="green", fontsize=12)
ax2.tick_params(axis="y", labelcolor="green")
ax2.legend(loc="upper right")

plt.title("Training Progress", fontsize=14)
plt.tight_layout()
plt.show()
