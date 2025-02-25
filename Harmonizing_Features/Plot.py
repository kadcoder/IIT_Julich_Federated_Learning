import pickle
import os
import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split


# Function to extract epoch number from file name to eventually sort it 
def extract_epoch_number(file_path):
    match = re.search(r"Epoch(\d+)", file_path)
    return int(match.group(1)) if match else float('inf')  # In case no "Epoch" number is found


def test_loss_graph(test_loss_folder,plot_name="Global_Test_Loss"): #path to folder containing the pickle files
    # Initialize dictionaries to store losses
    losses_over_epochs = {'eNki': [], 'SALD': [], 'CamCAN': []}
    loss_paths = glob.glob(os.path.join(test_loss_folder, "*"))

    # Sort the files based on epoch numbers extracted from file names
    loss_paths.sort(key=extract_epoch_number)
    
        # Check if there are files and load the second one (index 1)
    if loss_paths:
            pickle_file_path = loss_paths[-1]  # Get the second file in the sorted list

            # Load and print the pickle file contents
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
                print(f"Data from second file ({pickle_file_path}):")
                print(data)
    else:
        print("No files found in the directory.")
    
    
    loss_list = []

    # Iterate through the sorted pickle files and process them
    for loss_file in loss_paths:
        with open(loss_file, "rb") as f:
            test_losses = pickle.load(f)
            loss_list.append(test_losses)

    # Extracting keys from the first pickle file
    dict_keys = list(loss_list[0].keys())

    # Organize losses over epochs for each key
    for losses in loss_list:
        for key in dict_keys: 
            losses_over_epochs[key].append(losses[key])

    # Plotting
    plt.figure(figsize=(12, 6))
        
    # Plot each loss series with a different color
    plt.plot(losses_over_epochs['eNki'], label='eNki', color='b')  
    plt.plot(losses_over_epochs['SALD'], label='SALD', color='r')  
    plt.plot(losses_over_epochs['CamCAN'], label='CamCAN', color='g')  

    # Customize the plot
    plt.title("Loss over FedEpochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True)

    plt.savefig(plot_name+".png", dpi=300, bbox_inches='tight')
    plt.close()
     

        
def local_loss_graphs(local_loss_folder, plot_name="local_test_loss",site_name="local"):
    local_paths = glob.glob(os.path.join(local_loss_folder, "*.pkl"))  # Only select .pkl files

    # Sort file paths based on extracted epoch numbers
    local_paths.sort(key=extract_epoch_number)

    train_list = []
    validate_list = []
    test_list = []

    for loss_file in local_paths:
        loss_list = []
        with open(loss_file, 'rb') as f:
            data = pickle.load(f)
            loss_list.append(data)  
        train_list.append(loss_list[0]['training losses'])
        validate_list.append(loss_list[0]['validate losses'])
        test_list.append(loss_list[0]['test loss'])   

    # Flatten lists if needed
    train_list_merged = [item for sublist in train_list for item in sublist]
    validate_list_merged = [item for sublist in validate_list for item in sublist]
    test_list_merged = [item for sublist in test_list for item in sublist]

    num_epochs = len(train_list_merged)

    # Determine x-ticks at every 1/4th of the total epochs
    xticks_positions = list(range(0, len(train_list_merged), 20))

    plt.figure(figsize=(12, 6))
    plt.plot(train_list_merged, label="train", color='b')
    plt.plot(validate_list_merged, label="validate", color='r') 
    plt.plot(test_list_merged, label="test", color='g')   

    plt.xticks(xticks_positions)  # Apply the calculated x-ticks

    plt.title("Combined loss over multiple Fed epochs for "+str(site_name)) 
    plt.xlabel("Total Epochs")
    plt.ylabel("Loss")  
    plt.grid(True)
    plt.legend(loc="best") 
    plt.savefig(plot_name + ".png", dpi=300, bbox_inches='tight')
    plt.close()
    
"-------------------------------------------------------final execution---------------------------------------------------------------"

test_loss_folder=r"C:\Users\deoku\OneDrive\Documents\kunal\IIT_work\Julich_IIT_Collab\Test_loss_Parameters_Local_10_Fed_60_SGD"
test_loss_graph(test_loss_folder)

local_loss_folder1=r"C:\Users\deoku\OneDrive\Documents\kunal\IIT_work\Julich_IIT_Collab\Federated_Parameters_Local_10_Fed_60_SGD\CamCAN"
local_loss_folder2=r"C:\Users\deoku\OneDrive\Documents\kunal\IIT_work\Julich_IIT_Collab\Federated_Parameters_Local_10_Fed_60_SGD\SALD"
local_loss_graphs(local_loss_folder1,"Local1_Test_Loss","CamCAN")
local_loss_graphs(local_loss_folder2,"Local2_Test_Loss","SALD")
