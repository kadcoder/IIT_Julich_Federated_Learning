import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import NAdam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AgePredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3,LeakyReLu_slope=0.01):
        super(AgePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLu_slope)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leakyrelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Custom Dataset for PyTorch
class AgePredictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(df,target_column):
    '''
    df: Input dataframe 
    target_column: Column containing target variables
    '''
    # Separate features and labels
    X = df.drop(columns=[target_column])  # Assuming 'age' is the target column
    y = df[target_column]

    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42) #original test_size=0.2
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42) #original test_size=0.4

    # Normalize the features (scaling them to have 0 mean and 1 standard deviation)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def train_model(model, X_train, X_val, X_test, y_train, y_val, y_test, epochs=500, learning_rate=0.00005, device=None, weights_path='best_model_weights.pth'):
    model = model.to(device)
    
    train_dataset = AgePredictionDataset(X_train, y_train)
    val_dataset = AgePredictionDataset(X_val, y_val)
    test_dataset = AgePredictionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

    train_losses, val_losses = [], []
    patience_model_reload = 25
    best_val_loss = float('inf')
    epochs_without_improvement_model_reload = 0
    best_model_weights = None
    gradients_per_epoch = {} #Dictionary used to collect epoch-wise gradients 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        batch_gradients = {}

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in batch_gradients:
                        batch_gradients[name] = []
                    batch_gradients[name].append(param.grad.clone().detach()) #append layer-wise gradients to this dictionary for each epoch

            optimizer.step()

        gradients_per_epoch[epoch] = {   
            name: torch.mean(torch.stack(batch_grads), dim=0)
            for name, batch_grads in batch_gradients.items()
        }

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train MAE Loss: {avg_train_loss:.4f}, Val MAE Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, weights_path)
            epochs_without_improvement_model_reload = 0
        
        else:
            epochs_without_improvement_model_reload += 1

        if epochs_without_improvement_model_reload >= patience_model_reload and best_model_weights is not None:
            print("Restoring the best model weights due to early stopping.")
            model.load_state_dict(torch.load(weights_path))
            epochs_without_improvement_model_reload = 0

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test MAE Loss: {avg_test_loss:.4f}")

    return train_losses, val_losses, avg_test_loss, gradients_per_epoch

def weighted_avg_gradients(gradients1, gradients2, weight1=0.5, weight2=0.5):
    """
    Compute the weighted average of two sets of gradients stored as dictionaries.

    Parameters:
    gradients1 (dict): First set of gradients in the format {epoch: {layer: tensor_values}}.
    gradients2 (dict): Second set of gradients in the same format.
    weight1 (float): Weight for the first set of gradients.
    weight2 (float): Weight for the second set of gradients.

    Returns:
    dict: Weighted average of the gradients in the same dictionary format.
    """
    weighted_gradients = {}

    # Ensure both gradient dictionaries have the same structure
    if gradients1.keys() != gradients2.keys():
        raise ValueError("Gradient dictionaries must have the same epochs.")

    for epoch in gradients1.keys():
        if gradients1[epoch].keys() != gradients2[epoch].keys():
            raise ValueError(f"Gradient dictionaries must have the same layers at epoch {epoch}.")

        # Compute weighted average for each layer in the epoch
        weighted_gradients[epoch] = {}
        for layer in gradients1[epoch].keys():
            g1 = gradients1[epoch][layer]
            g2 = gradients2[epoch][layer]

            # Ensure tensors are on CPU before converting to numpy
            g1 = g1.cpu().numpy() if g1.is_cuda else g1.numpy()
            g2 = g2.cpu().numpy() if g2.is_cuda else g2.numpy()

            if g1.shape != g2.shape:
                raise ValueError(f"Gradients for layer '{layer}' must have the same shape at epoch {epoch}.")

            # Compute weighted average
            weighted_gradients[epoch][layer] = weight1 * g1 + weight2 * g2

    return weighted_gradients

def federated_learning(global_model, silo1_data, silo2_data, test_paths, dataset_names, weights_path='best_weights_global_model.pth', fed_epochs=10, local_epochs=350, lr=0.00005, device=None):
    gm = global_model.to(device)
    #model1 = deepcopy(gm)
    #model2 = deepcopy(gm)
    parameter_list = {}
    loss_dict = {i: {} for i in range(fed_epochs)}  # Initialize the loss_dict with empty dictionaries for each federated epoch
    
    for fed_epoch in range(fed_epochs):
        # Create deep copies of the global model for each silo
        model1 = deepcopy(gm)
        model2 = deepcopy(gm)
        
        temp_list = []  # format parameter_list------->temp_list------>model parameters
        print("This is federated epoch number ", str(fed_epoch + 1))
        
        # For Silo1
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = preprocess_data(silo1_data, 'age')
        train_loss1, validate_loss1, test_loss1, gradients1 = train_model(model1, X_train1, X_val1, X_test1, y_train1, y_val1, y_test1, local_epochs, lr, device, weights_path='Dump1.pth')
        Silo1_parameters = {"training losses": train_loss1, "validate losses": validate_loss1, "test loss": test_loss1}  # Save Silo1 model parameters
        temp_list.append(Silo1_parameters)  # append Silo1 parameters to temp_list
        
        # For Silo2
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = preprocess_data(silo2_data, 'age')
        train_loss2, validate_loss2, test_loss2, gradients2 = train_model(model2, X_train2, X_val2, X_test2, y_train2, y_val2, y_test2, local_epochs, lr, device, weights_path='Dump2.pth')
        Silo2_parameters = {"training losses": train_loss2, "validate losses": validate_loss2, "test loss": test_loss2}  # Save Silo2 model parameters
        temp_list.append(Silo2_parameters)  # append Silo2 parameters to temp_list
        
        averaged_gradients = weighted_avg_gradients(gradients1, gradients2)
        
        # Initialize the optimizer for the global model (you can use any optimizer, here we use SGD)
        optimizer = torch.optim.Adam(gm.parameters(), lr=lr, weight_decay=1e-4)

        # For each epoch, apply the averaged gradients using the optimizer
        for epoch in range(local_epochs):
            optimizer.zero_grad()  # Zero out any existing gradients in the optimizer
            
            with torch.no_grad():
                epoch_gradients = averaged_gradients.get(epoch, {})
                
                for layer_name, grad in epoch_gradients.items():
                    grad_tensor = torch.tensor(grad).to(device) if not isinstance(grad, torch.Tensor) else grad.to(device)
                    
                    param = dict(gm.named_parameters()).get(layer_name)
                    if param is not None:
                        # Set the gradients for the parameter (this is how the optimizer knows what to update)
                        param.grad = grad_tensor
                
                optimizer.step()  # Apply the gradients and update the model parameters


        # Save the updated global model weights
        torch.save(gm.state_dict(), weights_path)
        
        parameter_list[fed_epoch] = temp_list
        
        # Test the model on different datasets and save the test losses for each site
        for test_path, dataset_name in zip(test_paths, dataset_names):
            df_unseen = pd.read_csv(test_path) #Dataframe for unseen test data

            # Separate features and target
            X_test = df_unseen.drop(columns=['age'])
            y_test = df_unseen['age']

            # Scale the features using StandardScaler
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)  # Scaled features as NumPy array

            # Convert X_test to tensor
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            # Convert y_test to tensor (reshaped to (num_samples, 1))
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

            # Create DataLoader
            test_dataset = AgePredictionDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Ensure model is in eval mode
            gm.eval()

            # Evaluate the model (example using L1 Loss)
            criterion = nn.L1Loss()  # Mean Absolute Error for regression
            total_test_loss = 0

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass through the model
                outputs = gm(inputs)
                
                # Calculate the loss
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

            # Average test loss
            average_test_loss = total_test_loss / len(test_loader)

            # Store the test loss for each federated epoch and dataset (site)
            loss_dict[fed_epoch][dataset_name] = round(average_test_loss, 4)

    return gm, parameter_list, loss_dict




'''-------------------------------------------------Final Execution starts here---------------------------------------'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global data eNKI Silo1 SALD and Silo2 CamCAN
df_global= pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/eNKI/Train_eNki_Stratified.csv")
silo1=pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/SALD/Harmonized_Train_SALD_Stratified.csv")
silo2=pd.read_csv("/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/CamCAN/Harmonized_Train_CamCAN_Stratified.csv")

test_paths=["/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/eNKI/Test_eNki_Stratified.csv",
                "/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/SALD/Test_SALD_Stratified.csv",
                "/home/kunaldeo/Julich_IIT_Collab/data/brain_age_parcels/CamCAN/Test_CamCAN_Stratified.csv"] #Define paths to test sets

dataset_names=["eNki","SALD","CamCAN"] #pass names of datasets

global_model_state= 'uninitialized' #Select whether global model is initialized or uninitialized
global_model_final=None
parameter_list=None

#assign paths to save the weights of global models
weights_path="/home/kunaldeo/Julich_IIT_Collab/best_weights_global_model_stratified.pth"

local_epochs=100
fed_epochs=10

#Initial model and preprocess data
X_train_global, X_val_global, X_test_global, y_train_global, y_val_global, y_test_global = preprocess_data(df_global,'age')
global_model=AgePredictor(input_size=X_train_global.shape[1], dropout_rate=0.3)

if global_model_state == 'uninitialized':
   
    global_model_final, parameter_list, loss_dict = federated_learning(
    global_model, 
    silo1, 
    silo2, 
    test_paths=test_paths,  # List of test dataset paths
    dataset_names=dataset_names,  # Corresponding names of the datasets
    device=device, 
    weights_path=weights_path,
    local_epochs=local_epochs, 
    fed_epochs=fed_epochs
)
    
elif global_model_state == 'initialized':
    train_model(global_model,X_train_global,X_val_global,X_test_global,y_train_global,y_val_global,y_test_global,epochs=100,weights_path='Dump.pth')
   
    global_model_final, parameter_list, loss_dict = federated_learning(
    global_model, 
    silo1, 
    silo2, 
    test_paths=test_paths,  # List of test dataset paths
    dataset_names=dataset_names,  # Corresponding names of the datasets
    device=device, 
    weights_path=weights_path, 
    local_epochs=local_epochs, 
    fed_epochs=fed_epochs
)
    
for i in parameter_list:
    #Save losses for SALD local training as a pickle file
    with open("/home/kunaldeo/Julich_IIT_Collab/Federated_Parameters/SALD_parameters_Fed_Epoch"+str(i)+".pkl", "wb") as f:
      pickle.dump(parameter_list[i][0], f)
    #Save losses for CamCAN local training as a pickle file
    with open("/home/kunaldeo/Julich_IIT_Collab/Federated_Parameters/CamCAN_parameters_Fed_Epoch"+str(i)+".pkl", "wb") as f:
      pickle.dump(parameter_list[i][1], f)
     

    # Save the test losses at the end of all federated epochs
    with open("/home/kunaldeo/Julich_IIT_Collab/Test_loss_Parameters/Test_losses_Fed_Epoch" + str(i) +".pkl", "wb") as f:
      pickle.dump(loss_dict[i], f)
        
