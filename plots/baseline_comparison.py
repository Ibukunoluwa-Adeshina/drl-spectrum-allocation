import matplotlib.pyplot as plt
import os

# create results directory if missing
os.makedirs("results", exist_ok=True)

# actual results
methods = [
    "Random",
    "Greedy",
    "DQN"
]

rewards = [
    199.04,
    178.98,
    207.77
]

# create figure
plt.figure(figsize=(8,5))

# create bars
bars = plt.bar(methods, rewards)

# labels
plt.xlabel("Method")
plt.ylabel("Average Reward")
plt.title("Spectrum Allocation Performance Comparison")

# grid
plt.grid(True)

# annotate values
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

# force layout
plt.tight_layout()

# save figure
plt.savefig(
    "results/baseline_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

print("Plot saved successfully.")

# display
plt.show(block=True)