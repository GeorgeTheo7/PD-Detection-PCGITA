import torch
import os

########Εδώ ορίζω όλα τα paths και τα hyperparameters

#PATHS
EXPERIMENT_ROOT = "/mnt/datadrive/gtheodosiou/projects/thesis1/PCGITA_Experiment"

#Data paths
DATA_DIR = os.path.join(EXPERIMENT_ROOT, "data", "processed")
CSV_PATH = os.path.join(DATA_DIR, "pcgita_combined_5s_rich.csv") #PCGITA combined csv containing all data info
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks", "chunks")

#Saving Results
SAVE_MODEL_DIR = os.path.join(EXPERIMENT_ROOT, "models_3channel")
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

#HYPERPARAMETERS
TARGET_SR = 16000 #all audio will be resampled to 16kHz
CHUNK_SEC = 5.0 
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC) #the expected number of samples in each chunk
MAX_LEN = 80000

BATCH_SIZE = 64 #number of samples per training batch
LEARNING_RATE = 1e-4
N_EPOCHS = 15
N_FOLDS = 5 #5 fold cross-validation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_WORKERS = 4
IMG_SIZE = (224, 224) #input image size for ResNet (resnet was trained on 224x224 images)