from src.models import model_packages
from src.pipeline import CrossValPipeline

pipeline = CrossValPipeline()

def test_cross_validate_models():
    for model_package in model_packages:
        pipeline.tune_hyperparameters(model_package)
        avg_f1 = pipeline.cross_validate_optimized_model(model_package)
        print(avg_f1)