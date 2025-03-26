from training import train_centralmodel
from evaluation import evaluate_centralmodel
import os

def experiment_OA(base_dir):

    silo_data_path = os.path.join(base_dir, 'silo_Datasets')
    for silo in ['CamCAN', 'SALD', 'eNki']:

        f = open(f'{silo}_results.txt', 'w')
        train_path = os.path.join(base_dir,f'{silo}/Train_{silo}.csv')
        test_path = os.path.join(silo_data_path,f'{silo}/Test_{silo}.csv')
        
        for max_epochs in [5, 10, 20, 30, 50, 80, 100, 150, 200]:
            scaler, model = train_centralmodel(train_path, max_epochs)
            avg_loss = evaluate_centralmodel(test_path,model,scaler)
            f.write(f'Max epochs: {max_epochs} and test MAE error: {avg_loss}\n')

        f.close()

if __name__ == "__main__":
    base_dir = input("Enter the base directory: ")
    experiment_OA(base_dir)
