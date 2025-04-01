# %%
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# adding the path of the parent directory
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # noqa
sys.path.append(project_dir)
from lib.training import train_centralmodel         # noqa
from lib.evaluation import evaluate_centralmodel    # noqa
from lib.data_utils import preprocess               # noqa
# %%

# Set the maximum number of epochs for training
max_epochs = 200
# Set the path to the directory containing the silo datasets
silo_data_path = os.path.join(project_dir, 'silo_Datasets')

# TODO: change eNki for eNKI
for silo in ['CamCAN', 'SALD', 'eNki']:
    # Store losses for plotting
    train_losses_dict = {}
    val_losses_dict = {}

    # TODO: Save results as .csv
    f = open(f'{silo}_results.txt', 'w')
    train_path = os.path.join(silo_data_path, f'{silo}/Train_{silo}.csv')
    test_path = os.path.join(silo_data_path, f'{silo}/Test_{silo}.csv')

    train_losses, val_losses, plotting_scaler, best_model = train_centralmodel(train_path, max_epochs)        # noqa

    scaler = StandardScaler()
    X_train, _, _ = preprocess(train_path)
    scaler.fit(X_train)

    avg_loss = evaluate_centralmodel(test_path, best_model, scaler)
    f.write(f'Max epochs: {max_epochs} and test MAE error: {avg_loss}\n')

    f.close()
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"Train Loss ({max_epochs} epochs)")                  # noqa
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"Val Loss ({max_epochs} epochs)", linestyle='dashed')    # noqa

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses for Different Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"OA_{silo}_losses.png")
    plt.close()
