"""Central configuration: audio/feature settings, model and training hyperparameters.

Paths are NOT hardcoded here; they are passed on the command line (see main.py).
"""

# ---------------- Audio / chunking ----------------
TARGET_SR = 16_000            # all audio resampled to 16 kHz (required by Wav2Vec2)
CHUNK_SEC = 5.0               # non-overlapping 5 s chunks
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)   # 80,000 samples
MIN_VALID_SEC = 2.5           # trailing chunks shorter than this are dropped

# PC-GITA tasks used (connected speech only)
TASKS = {
    "read_text": "read text/ayerfuialmedico/sin normalizar",
    "monologue": "monologue/sin normalizar",
}
CLASS_DIRS = {"hc": 0, "pd": 1}  # sub-folder name -> label (0 = healthy control, 1 = Parkinson's)

# Speaker ID at the start of every PC-GITA filename, e.g. AVPEPUDEA0001 (PD) / AVPEPUDEAC0001 (HC)
SUBJECT_ID_REGEX = r"^(AVPEPUDEAC?\d{4})"

# ---------------- Features ----------------
N_FFT = 1024
WIN_LENGTH = int(0.040 * TARGET_SR)   # 40 ms window
HOP_LENGTH = int(0.010 * TARGET_SR)   # 10 ms hop
N_MELS = 200
N_MFCC = 40
IMG_SIZE = (224, 224)                 # ResNet-18 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WAV2VEC2_ID = "facebook/wav2vec2-large-xlsr-53-spanish"   # frozen, used as a feature extractor

# ---------------- Training (thesis protocol) ----------------
N_FOLDS = 5             # outer speaker-independent folds
ES_FRAC = 0.15          # fraction of training speakers held out for early stopping
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15
PATIENCE = 3            # epochs without improvement on the early-stopping set
GRAD_CLIP = 1.0
DECISION_THRESHOLD = 0.5
DEFAULT_SEEDS = [42, 1, 2, 3, 4]
