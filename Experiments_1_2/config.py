import torch,numpy as np

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global constants
BATCH_SIZE = 32
INIT_LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2
PATIENCE = 10
MAX_EPOCHS = [10,20,30,50] 
NUM_EPOCHS = 1
train_ratio = 0.7
MOMENTUM = 0.8
GRADIENT_CLIP = 1.0
GLOBAL_SCALER = None