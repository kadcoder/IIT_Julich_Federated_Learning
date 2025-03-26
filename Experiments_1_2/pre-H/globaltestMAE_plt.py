import re
import matplotlib.pyplot as plt

# Prepare containers for the data.
results = {
    "CamCAN": {"epochs": [], "test": []},
    "SALD": {"epochs": [], "test": []},
    "eNki": {"epochs": [], "test": []}
}

# Read file content
with open('results_pre-H_l15.txt', 'r') as file:
    lines = file.read().strip().splitlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue

    match = re.search(r"Test on (\w+) \| epochs: (\d+) \| Test MAE error: ([\d\.Ee+-]+)", line)
    if match:
        dataset = match.group(1)
        max_epochs = int(match.group(2))
        test_mae = float(match.group(3))

        if dataset in results:
            results[dataset]["epochs"].append(max_epochs)
            results[dataset]["test"].append(test_mae)

    i += 1

# Create subplots for the three datasets.
fig, ax = plt.subplots(figsize=(10, 6))

# Plot for each dataset.
for dataset in results.keys():
    ax.plot(results[dataset]["epochs"], results[dataset]["test"], label=f"{dataset} Test MAE", marker='o')

ax.set_title("Test MAE Error vs Global Epochs(pre-H)")
ax.set_xlabel("Global Epochs")
ax.set_ylabel("Test MAE Error")
ax.set_ylim(0, 30)
ax.legend()
ax.grid(True)

plt.savefig('pre-H_Test_Global_MAE.png')
