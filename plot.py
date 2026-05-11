import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/training_log.csv")

# Calculate the actual total ELBO loss: Recon + Beta * KL
df["total_loss"] = df["train_loss"] + df["beta"] * df["kl_loss"]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot train_loss (recon), total_loss, and val_loss on the primary y-axis
ax1.plot(df["step"], df["train_loss"], label="Train Loss (Recon)", color="blue", linewidth=2)
ax1.plot(df["step"], df["total_loss"], label="Total Loss (Recon + Beta*KL)", color="red", linewidth=2, linestyle="-.")
ax1.plot(df["step"], df["val_loss"], label="Val Loss", color="orange", linewidth=2)
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Reconstruction Loss", color="black", fontsize=12)
ax1.tick_params(axis="y", labelcolor="black")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

# Plot kl_loss on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df["step"], df["kl_loss"], label="KL Loss", color="green", linestyle="--", linewidth=2)
ax2.set_ylabel("KL Loss", color="green", fontsize=12)
ax2.tick_params(axis="y", labelcolor="green")
ax2.legend(loc="upper right")

plt.title("Training Progress", fontsize=14)
plt.tight_layout()
plt.show()
