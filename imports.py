import pandas  as pd
import numpy as np
np.random.seed(42)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from pathlib import Path
import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
from tabpfn import TabPFNClassifier