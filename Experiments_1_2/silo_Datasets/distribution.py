import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


path = '/home/tanurima/germany/brain_age_parcels/kunal/Filtered_Datasets'
# Load data from CSV files
def load_data(file_path):
    return pd.read_csv(file_path).select_dtypes(include=np.number)

# File paths (replace with actual paths)
file_paths = {}
for silo in ['CamCAN','eNki','SALD']:
    file_paths[silo] = os.path.join(path,f'{silo}/Train_{silo}.csv') #.replace('train','test')

#file_paths = ["file1.csv", "file2.csv", "file3.csv"]

# Reduce subset size further for efficiency
subset_size = 400  # Limit each dataset to 500 samples
# Load the full datasets
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Identify common numerical columns across datasets
common_columns = set(dataframes["eNki"].select_dtypes(include=np.number).columns)
for df in dataframes.values():
    common_columns.intersection_update(df.select_dtypes(include=np.number).columns)

common_columns = list(common_columns)

# Extract and standardize the common features
# Reduce dataset size for faster t-SNE processing
subset_size = 400  # Limit each dataset to 1000 samples (if possible)
X = np.concatenate([df[common_columns].values[:subset_size] for df in dataframes.values()], axis=0)
y = np.concatenate([[name] * min(subset_size, len(df)) for name, df in dataframes.items()])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 6))
for name, color in zip(file_paths.keys(), ['red', 'blue', 'green']):
    indices = (y == name)
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=name, alpha=0.6, color=color)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of train Dataset Distributions")
plt.legend()
#plt.show()

plt.savefig('tsne-silos-tr-distribution.png')

############################    Harmonized dataset  ################

# File paths (replace with actual paths)
file_paths = {}
for silo in ['CamCAN','eNki','SALD']:
    if silo == 'eNki':
        file_paths[silo] = os.path.join(path,f'{silo}/Train_{silo}.csv')
    else:
        file_paths[silo] = os.path.join(path,f'{silo}/Train_harmonized_{silo}.csv')

#file_paths = ["file1.csv", "file2.csv", "file3.csv"]

# Reduce subset size further for efficiency
subset_size = 400  # Limit each dataset to 500 samples
# Load the full datasets
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Identify common numerical columns across datasets
common_columns = set(dataframes["eNki"].select_dtypes(include=np.number).columns)
for df in dataframes.values():
    common_columns.intersection_update(df.select_dtypes(include=np.number).columns)

common_columns = list(common_columns)

# Extract and standardize the common features
# Reduce dataset size for faster t-SNE processing
subset_size = 400  # Limit each dataset to 1000 samples (if possible)
X = np.concatenate([df[common_columns].values[:subset_size] for df in dataframes.values()], axis=0)
y = np.concatenate([[name] * min(subset_size, len(df)) for name, df in dataframes.items()])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 6))
for name, color in zip(file_paths.keys(), ['red', 'blue', 'green']):
    indices = (y == name)
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=name, alpha=0.6, color=color)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of train harmonized Dataset Distributions")
plt.legend()
#plt.show()

plt.savefig('tsne-silos-harmo-tr-distribution.png')
