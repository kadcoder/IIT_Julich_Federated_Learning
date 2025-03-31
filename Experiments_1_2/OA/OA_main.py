import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import train_centralmodel
from evaluation import evaluate_centralmodel
import os

def experiment_OA(base_dir):

    silo_data_path = os.path.join(base_dir, 'silo_Datasets')
    for silo in ['CamCAN', 'SALD', 'eNki']:
        # Store losses for plotting
        train_losses_dict = {}
        val_losses_dict = {}

        f = open(f'{silo}_results.txt', 'w')
        train_path = os.path.join(silo_data_path,f'{silo}/Train_{silo}.csv')
        test_path = os.path.join(silo_data_path,f'{silo}/Test_{silo}.csv')
        
        for max_epochs in [5, 10, 20, 30, 50, 80, 100, 150, 200]:
            train_losses,val_losses,scaler, model = train_centralmodel(train_path, max_epochs)
            # Store losses with epoch count as key
            train_losses_dict[max_epochs] = train_losses
            val_losses_dict[max_epochs] = val_losses

            avg_loss = evaluate_centralmodel(test_path,model,scaler)
            f.write(f'Max epochs: {max_epochs} and test MAE error: {avg_loss}\n')

        f.close()
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))

        for max_epochs in train_losses_dict:
            plt.plot(range(1, len(train_losses_dict[max_epochs]) + 1), train_losses_dict[max_epochs], label=f"Train Loss ({max_epochs} epochs)")
            plt.plot(range(1, len(val_losses_dict[max_epochs]) + 1), val_losses_dict[max_epochs], label=f"Val Loss ({max_epochs} epochs)", linestyle='dashed')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses for Different Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"OA_{silo}_losses.png")
        plt.close()

if __name__ == "__main__":
    #base_dir = input("Enter the base directory: ")
    base_dir ="/home/tanurima/germany/brain_age_parcels/Experiments_1_2"
    experiment_OA(base_dir)
