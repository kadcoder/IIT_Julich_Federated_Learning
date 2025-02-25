import torch

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)

# Global flag to track initialization
WEIGHTS_INITIALIZED = False
DROPOUT_RATE = 0.3

# Global scaler should be computed once
GLOBAL_SCALER = None

# Hyperparameters
BATCH_SIZE = 32
INIT_LR = 5e-4
L1_LAMBDA = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MAX_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
NUM_EPOCHS = 20
train_ratio = 0.6
import warnings
warnings.filterwarnings('ignore')