from imports import np, CatBoostClassifier
from constants import RANDOM_SEED, EARLY_STOPPING_ROUNDS
from src.models import prediction_packages
from src.pipeline import PredictionPipeline

def test_make_predictions():
    pipeline = PredictionPipeline()
    
    for package in prediction_packages:
        pipeline.make_testing_preds(package)
        break