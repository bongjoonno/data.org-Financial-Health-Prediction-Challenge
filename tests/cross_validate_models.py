from src.models import model_packages
from src.pipeline import CrossValPipeline

pipeline = CrossValPipeline()

def test_cross_validate_models():
    for model_package in model_packages:
        avg_f1 = pipeline.tune_hyperparameters(model_package)
        print(avg_f1)