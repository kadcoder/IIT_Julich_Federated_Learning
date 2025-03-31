import pandas as pd,os,sys,matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import train_centralmodel
from evaluation import evaluate_centralmodel

def experiment_OB(base_dir):
    silos = ['CamCAN', 'SALD']
    silo_data_path = os.path.join(base_dir, 'silo_Datasets')
    train_files = [os.path.join(silo_data_path,f'{silo}/Train_{silo}.csv') for silo in silos]
    test_files = {silo: os.path.join(silo_data_path,f'{silo}/Test_{silo}.csv') for silo in silos + ['eNki']}
    
    combined_train_df = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)

    f = open('OB_results.txt', 'w')
    # Store losses for plotting
    train_losses_dict = {}
    val_losses_dict = {}

    for max_epochs in [50, 80, 100, 150]:
        train_losses,val_losses,scaler,model = train_centralmodel(combined_train_df, max_epochs)

        # Store losses with epoch count as key
        train_losses_dict[max_epochs] = train_losses
        val_losses_dict[max_epochs] = val_losses

        for silo, test_path in test_files.items():
            avg_loss = evaluate_centralmodel(test_path,model,scaler)
            f.write(f'Test on {silo} | Max epochs: {max_epochs} | Test MAE error: {avg_loss}\n')
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
    plt.savefig("OB_losses.png")
    plt.close()

if __name__ == "__main__":
    #base_dir = input("Enter the base directory: ")
    base_dir ="/home/tanurima/germany/brain_age_parcels/Experiments_1_2"
    experiment_OB(base_dir)
