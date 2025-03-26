import pandas as pd,os
from training import train_centralmodel
from evaluation import evaluate_centralmodel

def experiment_OB(base_dir):
    silos = ['CamCAN', 'SALD']
    silo_data_path = os.path.join(base_dir, 'silo_Datasets')
    train_files = [os.path.join(silo_data_path,f'{silo}/Train_{silo}.csv') for silo in silos]
    test_files = {silo: os.path.join(silo_data_path,f'{silo}/Test_{silo}.csv') for silo in silos + ['eNki']}
    
    combined_train_df = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)

    f = open('OB_results.txt', 'w')

    for max_epochs in [50, 80, 100, 150]:
        scaler,model = train_centralmodel(combined_train_df, max_epochs)
        for silo, test_path in test_files.items():
            avg_loss = evaluate_centralmodel(test_path,model,scaler)
            f.write(f'Test on {silo} | Max epochs: {max_epochs} | Test MAE error: {avg_loss}\n')
    f.close()

if __name__ == "__main__":
    base_dir = input("Enter the base directory: ")
    experiment_OB(base_dir)
