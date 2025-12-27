from imports import Path

STRING_TO_NUM_TARGET = {'Low' : 0, 'Medium' : 1, 'High' : 2}
NUM_TO_STRING_TARGET = {v: k for k, v in STRING_TO_NUM_TARGET.items()}

DATA_PATH = Path(r'D:\code\repos\data.org-Financial-Health-Prediction-Challenge\data')
RANDOM_SEED = 42
MAX_EPOCHS = 5_000
DEFAULT_LR = 0.05
EARLY_STOPPING_ROUNDS = 100
DEFAULT_TREE_DEPTH = 6
LEARNING_RATES = 50
DEFAULT_THRESHOLDS = [0.5, 0.5, 0.5]