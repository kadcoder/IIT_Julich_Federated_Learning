#%%
import pandas as pd
import csv
import os
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# adding the path of the parent directory
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # noqa
sys.path.append(project_dir)
from lib.training import train_centralmodel                                                         # noqa
from lib.evaluation import evaluate_centralmodel,predict_age                                        # noqa
from lib.data_utils import preprocess                                                               # noqa
from lib.utils import ensure_dir                                                                   # noqa
#%%

# Set the maximum number of epochs for training
max_epochs = 200
# Set the path to the directory containing the silo datasets
silo_data_path = os.path.join(project_dir, 'silo_Datasets')
# Set the path to save results
results_csv_path = os.path.join(project_dir, 'Experiments_1_2/OB/results_OB')
os.makedirs(results_csv_path, exist_ok=True)  #Create directory if missing

silos = ['CamCAN', 'SALD','eNKI']
train_files_df = [os.path.join(silo_data_path,f'{silo}/Train_{silo}.csv') for silo in silos]

train_files = {silo: os.path.join(silo_data_path,f'{silo}/Train_{silo}.csv') for silo in silos}
test_files = {silo: os.path.join(silo_data_path,f'{silo}/Test_{silo}.csv') for silo in silos}

combined_train_df = pd.concat([pd.read_csv(file) for file in train_files_df], ignore_index=True)
train_losses, val_losses, scaler, best_model = train_centralmodel(combined_train_df, max_epochs)
 
# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"Train Loss ({max_epochs} epochs)")
plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"Val Loss ({max_epochs} epochs)", linestyle='dashed')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses for Different Epochs")
plt.legend()
plt.grid(True)

# Ensure the directory for plots exists
plot_path = os.path.join(results_csv_path, f"OB_losses.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)  #Create directory if missing

# Save the plot
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.close()

#Save results as .csv
csv_filename_path = os.path.join(results_csv_path,'OB_results.csv')
os.makedirs(os.path.dirname(csv_filename_path), exist_ok=True)  #Create directory if missing
results_age_path = os.path.join(project_dir, 'Experiments_1_2/OB/results_OB/predictions')
ensure_dir(results_age_path)  # Ensure the directory exists

# Open the CSV file once for writing
with open(csv_filename_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(['Silo', 'Max Epochs', 'Test MAE Error'])
    
    for silo, train_path in train_files.items():
        scaler = StandardScaler()
        X_train, _, _ = preprocess(train_path)
        scaler.fit(X_train)
    
        # Evaluate the model on the test set
        test_path = test_files[silo]
        avg_loss = evaluate_centralmodel(test_path, best_model, scaler)
        predict_age(best_model, test_path, results_age_path ,scaler)
        
        # Write a row for the current silo
        writer.writerow([silo, max_epochs, avg_loss])