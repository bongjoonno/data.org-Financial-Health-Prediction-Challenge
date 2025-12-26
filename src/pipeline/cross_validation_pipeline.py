from imports import np, f1_score, CatBoostClassifier
from constants import RANDOM_SEED, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_TREE_DEPTH
from src.data_prep import clean_data, train_df, split_data_k_folds, scale_data
from src.prediction_creation import make_preds_with_thresholds
from src.optimizers import HyperParamOptimizer

hyperparam_optimizer = HyperParamOptimizer()

class CrossValPipeline:
    n_folds = 5
    
    @staticmethod
    def tune_hyperparameters(model_package: dict):
        optimize_epochs_and_lr = model_package['optimize_epochs_and_lr']
        optimize_tree_depth = model_package['optimize_tree_depth']
        optimize_thresh = model_package['optimize_thresh']
                
        if optimize_epochs_and_lr:
            hyperparam_optimizer.optimize_epochs_and_learning_rate(model_package)
        
        if optimize_tree_depth:
            hyperparam_optimizer.optimize_tree_depth(model_package)
        
        if optimize_thresh:
            hyperparam_optimizer.get_best_thresholds(model_package)
    
    def cross_validate_optimized_model(model_package: dict):
        best_epochs = model_package.get('best_epochs', DEFAULT_EPOCHS)
        best_lr = model_package.get('best_lr', DEFAULT_LR)
        best_tree_depth = model_package.get('best_tree_depth', DEFAULT_TREE_DEPTH)
        thresholds = model_package.get('thresholds', [0.5, 0.5, 0.5])
        
        optimized_model = CatBoostClassifier(iterations=best_epochs,
                                             learning_rate=best_lr,
                                             early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
                                             depth=best_tree_depth,
                                             loss_function='MultiClass',
                                             verbose=0,
                                             random_seed=RANDOM_SEED,
                                             thread_count=1)
        
        one_hot_encode_categoricals = model_package['one_hot_encode_categoricals']
        scale_x = model_package['scale_x']
                
        df = clean_data(train_df, one_hot_encode_categoricals)
            
        x = df.drop(columns='Target')
        y = df['Target']
            
        folds = split_data_k_folds(x, y, CrossValPipeline.n_folds)
        
        avg_f1s = []
        
        for fold in folds:
            x_train, y_train, x_val, y_val = fold

            if scale_x:
                x_train, x_val = scale_data(x_train, x_val)

            optimized_model.fit(x_train, y_train,
                      eval_set=(x_val, y_val))

            y_pred = optimized_model.predict(x_val, prediction_type='Probability')

            y_pred = make_preds_with_thresholds(y_pred, thresholds)

            f1_per_class = f1_score(y_val, y_pred, average=None)
            avg_f1s.append(np.mean(f1_per_class))
            
        return np.mean(avg_f1s)