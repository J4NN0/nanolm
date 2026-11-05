import os

# Dataset
DATASET_NAME = "TinyStoriesV2-GPT4"
TRAIN_TEXT_PATH = f"data/{DATASET_NAME}-train.txt"
VALID_TEXT_PATH = f"data/{DATASET_NAME}-valid.txt"

# Tokenizer
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

# Model
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0

# Training: tokens processed = batch size × steps × context length (32 × 5,000 × 256 ≈ 41M)
BATCH_SIZE = 32
MAX_ITERS = 5000
MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 200

TOKENIZERS_DIR = "tokenizers"
CHECKPOINTS_DIR = "checkpoints"
TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, f"{DATASET_NAME}_{VOCAB_SIZE}.json")
TRAIN_TOKENS_PATH = f"data/{DATASET_NAME}-train_{VOCAB_SIZE}.bin"
VALID_TOKENS_PATH = f"data/{DATASET_NAME}-valid_{VOCAB_SIZE}.bin"
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, f"{DATASET_NAME}_{VOCAB_SIZE}.pt")
