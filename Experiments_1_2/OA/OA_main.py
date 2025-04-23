# %%
import sys
import os
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# adding the path of the parent directory
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # noqa
sys.path.append(project_dir)
from lib.training import train_centralmodel                                                         # noqa
from lib.evaluation import evaluate_centralmodel,predict_age                                        # noqa
from lib.data_utils import preprocess                                                               # noqa
from lib.utils import ensure_dir                                                                    # noqa
# %%

# Set the maximum number of epochs for training
max_epochs = 200
# Set the path to the directory containing the silo datasets
silo_data_path = os.path.join(project_dir, 'silo_Datasets')
# Set the path to save results
results_csv_path = os.path.join(project_dir, 'Experiments_1_2/OA/results_OA')
os.makedirs(results_csv_path, exist_ok=True)  #Create directory if missing

for silo in ['CamCAN', 'SALD', 'eNKI']:
    # Store losses for plotting
    train_losses_dict = {}
    val_losses_dict = {}

    # Save results as .csv
    silo_results_csv_path = os.path.join(results_csv_path, f'{silo}_results.csv')   # noqa
    os.makedirs(os.path.dirname(silo_results_csv_path), exist_ok=True)  # Create directory if missing

    train_path = os.path.join(silo_data_path, f'{silo}/Train_{silo}.csv')
    test_path = os.path.join(silo_data_path, f'{silo}/Test_{silo}.csv')

    # Train the model
    train_losses, val_losses, plotting_scaler, best_model = train_centralmodel(train_path, max_epochs)  # noqa

    # Evaluate the model
    scaler = StandardScaler()
    X_train, _, _ = preprocess(train_path)
    scaler.fit(X_train)

    avg_loss = evaluate_centralmodel(test_path, best_model, scaler)
    predict_age(best_model, test_path, results_age_path, scaler)

    # TODO: there is no need to write row by row,
    # we can use pandas.to_csv to save the entire DataFrame
    # Not critical
    # Write results to CSV
    with open(silo_results_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])  # Header row
        for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), start=1):   # noqa
            writer.writerow([epoch, t_loss, v_loss])

    with open(test_results_csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row if the file is empty
        if csvfile.tell() == 0:
            writer.writerow(['Silo', 'Max Epochs', 'Test MAE'])
        # Write the results for this silo
        writer.writerow([silo, max_epochs, avg_loss])

    # TODO: Do the plots in another script
    # so we don't have to re-run all to generate them
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"Train Loss ({max_epochs} epochs)")                  # noqa
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"Val Loss ({max_epochs} epochs)", linestyle='dashed')    # noqa
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses for {silo}")
    plt.legend()
    plt.grid(True)

    # Ensure the directory for plots exists
    plot_path = os.path.join(results_csv_path, f"OA_{silo}_losses.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # noqa Create directory if missing

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")
    plt.close()
