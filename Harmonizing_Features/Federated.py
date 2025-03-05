import os
import pickle
import time
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from Age_Predictor import AgePredictor,AgePredictionDataset,Train_Model

class Federated():
    def federated_learning(global_model, silo1_data, silo2_data, test_paths, dataset_names, weights_path='best_weights_global_model.pth', fed_epochs=10, local_epochs=350, lr=0.00005, device=None):
        start_time = time.time()  # Start timer

        gm = deepcopy(global_model).to(device)
        parameter_list = {}
        loss_dict = {i: {} for i in range(fed_epochs)}  # Initialize loss dictionary for federated epochs
        
        for fed_epoch in range(fed_epochs):
            # Create deep copies of the global model for each silo
            model1 = deepcopy(gm).to(device) 
            model2 = deepcopy(gm).to(device)

            temp_list = []  
            print("Federated epoch:", fed_epoch + 1)

            # Preprocess Silo1 data
            X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = Train_Model.preprocess_data(silo1_data, 'age') #Get the Train Test Validation Sets for Silo1
            
            # Train model1 on silo1 data locally
            train_loss1, validate_loss1, test_loss1 = Train_Model.train_model(model1, X_train1, X_val1, X_test1, y_train1, y_val1, y_test1, local_epochs, lr, device)
            Silo1_parameters = {"training losses": train_loss1, "validate losses": validate_loss1, "test loss": test_loss1}
            temp_list.append(Silo1_parameters)

            # Preprocess Silo2 data
            X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = Train_Model.preprocess_data(silo2_data, 'age') #Get the Train Test Validation Sets for Silo2
            
            # Train model2 on silo2 data locally
            train_loss2, validate_loss2, test_loss2 = Train_Model.train_model(model2, X_train2, X_val2, X_test2, y_train2, y_val2, y_test2, local_epochs, lr, device)
            Silo2_parameters = {"training losses": train_loss2, "validate losses": validate_loss2, "test loss": test_loss2}
            temp_list.append(Silo2_parameters)

            # Apply average gradients to global model (gm) 
            optimizer = torch.optim.SGD(gm.parameters(), lr=lr, weight_decay=1e-3) #,momentum=0.1
            # Compute training set sizes
            n_samples1 = len(X_train1)  # Number of samples in Silo1
            n_samples2 = len(X_train2)  # Number of samples in Silo2
            total_samples = n_samples1 + n_samples2

            # Compute weights
            weight1 = n_samples1 / total_samples
            weight2 = n_samples2 / total_samples

            # Apply weighted averaged gradients
            for param1, param2, param3 in zip(model1.parameters(), model2.parameters(), gm.parameters()):
                if param1.grad is not None and param2.grad is not None:
                    avg_grad = weight1 * param1.grad + weight2 * param2.grad  # Weighted sum of gradients
                    param3.grad = avg_grad  # Assign averaged gradients to global model

            optimizer.step()

            # Save updated global model
            torch.save(gm.state_dict(), weights_path)
            parameter_list[fed_epoch] = temp_list

            # Evaluate on unseen test sets
            for test_path, dataset_name in zip(test_paths, dataset_names):
                df_unseen = pd.read_csv(test_path)
                X_test = df_unseen.drop(columns=['age'])
                y_test = df_unseen['age']
                scaler = StandardScaler()
                X_test = scaler.fit_transform(X_test)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
                test_dataset = AgePredictionDataset(X_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                gm.eval()
                criterion = nn.L1Loss()
                total_test_loss = 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = gm(inputs)
                    loss = criterion(outputs, targets)
                    total_test_loss += loss.item()

                loss_dict[fed_epoch][dataset_name] = round(total_test_loss / len(test_loader), 4)

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

        return gm, parameter_list, loss_dict



'''-------------------------------------------------Final Execution starts here---------------------------------------'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global data eNKI Silo1 SALD and Silo2 CamCAN
df_global= pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/eNKI/Train_eNki_Stratified.csv")
silo1=pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/SALD/Train_SALD_Stratified.csv")
silo2=pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/CamCAN/Train_CamCAN_Stratified.csv")

test_paths=["/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/eNKI/Test_eNki_Stratified.csv",
                "/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/SALD/Test_SALD_Stratified.csv",
                "/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/CamCAN/Test_CamCAN_Stratified.csv"] #Define paths to test sets

dataset_names=["eNki","SALD","CamCAN"] #pass names of datasets

global_model_state= 'uninitialized' #Select whether global model is initialized or uninitialized
global_model_final=None
parameter_list=None

#assign paths to save the weights of global models
weights_path="/home/kunaldeo/Julich_IIT_Collab/best_weights_global_model_stratified.pth"

#Initial model and preprocess global data in case of intitalization
X_train_global, X_val_global, X_test_global, y_train_global, y_val_global, y_test_global = Train_Model.preprocess_data(df_global,'age')
global_model=AgePredictor(input_size=X_train_global.shape[1],dropout_rate=0.3,LeakyReLu_slope=0.01)

if global_model_state == 'uninitialized':
   
    global_model_final, parameter_list, loss_dict = Federated.federated_learning(
    global_model, 
    silo1, 
    silo2, 
    test_paths=test_paths,  # List of test dataset paths
    dataset_names=dataset_names,  # Corresponding names of the datasets
    device=device, 
    lr=0.0012,  #for SGD lr=0.0012
    weights_path=weights_path,
    local_epochs=20, #for SGD= 20
    fed_epochs=1200 #for SGD =1200
)
    
elif global_model_state == 'initialized':
    Train_Model.train_model(global_model,X_train_global,X_val_global,X_test_global,y_train_global,y_val_global,y_test_global,epochs=20) #Keeping epochs same as local epochs
    global_model_final, parameter_list, loss_dict = Federated.federated_learning(
    global_model, 
    silo1, 
    silo2, 
    test_paths=test_paths,  # List of test dataset paths
    dataset_names=dataset_names,  # Corresponding names of the datasets
    device=device, 
    lr=0.0012,
    weights_path=weights_path,
    local_epochs=20, 
    fed_epochs=1200  
)   
else:
    print("Wrong input")    
        
# Define the base directories to save results
federated_results_CamCAN_dir = "/home/kunaldeo/Julich_IIT_Collab/Federated_Parameters/CamCAN" #Save Local dataset CamCAN losses    
federated_results_SALD_dir = "/home/kunaldeo/Julich_IIT_Collab/Federated_Parameters/SALD" #Save Local dataset SALD losses
test_loss_results_dir = "/home/kunaldeo/Julich_IIT_Collab/Test_loss_Parameters"

# Create directories if they do not exist
os.makedirs(federated_results_SALD_dir, exist_ok=True)
os.makedirs(federated_results_CamCAN_dir, exist_ok=True)
os.makedirs(test_loss_results_dir, exist_ok=True)

# Save the federated model parameters and test losses
for i in parameter_list:
    # Save SALD parameters
    sald_path = os.path.join(federated_results_SALD_dir, f"SALD_parameters_Fed_Epoch{i}.pkl")
    with open(sald_path, "wb") as f:
        pickle.dump(parameter_list[i][0], f)

    # Save CamCAN parameters
    camcan_path = os.path.join(federated_results_CamCAN_dir, f"CamCAN_parameters_Fed_Epoch{i}.pkl")
    with open(camcan_path, "wb") as f:
        pickle.dump(parameter_list[i][1], f)

    # Save test losses
    test_loss_path = os.path.join(test_loss_results_dir, f"Test_losses_Fed_Epoch{i}.pkl")
    with open(test_loss_path, "wb") as f:
        pickle.dump(loss_dict[i], f)
