import torch
from sklearn.preprocessing import StandardScaler

from brain_age_parcels.utils import utils
import config
import model as m

def evaluate_model(test_path, model_path):

    X_test,y_test,silo_name = utils.preprocess(test_path)
    # Load the model (assuming model_name corresponds to a model function or class)

    model = m.AgePredictor(X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))            # This looks quite subobtimal. Do you need to save the model every time that you evaluate it?. It makes more sense that the trained object is passed to the function
    model.eval()

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)    # This is wrong, you have to use the already trained standard scaler on the train data. You should pass the trained object to the function.

    # Convert to tensors
    test_loader = utils.dataloader(X_test,y_test)
    
    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.L1Loss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            y_batch = y_batch.view(-1, 1)
            predictions = model(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f'\nTest error of {silo_name} : {avg_loss}\n')
    return avg_loss
