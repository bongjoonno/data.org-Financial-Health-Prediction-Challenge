from imports import np, f1_score
from src.data_prep import clean_data, train_df, split_data_k_folds, scale_data
from src.prediction_creation import make_preds_with_thresholds
from src.optimizers import ThresholdOptimizer

threshold_optimizer = ThresholdOptimizer()

class CrossValPipeline:
    n_folds = 5
    
    @staticmethod
    def train_cross_val(model_package: dict):
        model = model_package['model']
        one_hot_encode_categoricals = model_package['one_hot_encode_categoricals']
        scale_x = model_package['scale_x']
        optimize_thresh = model_package['optimize_thresh']
            
        df = clean_data(train_df, one_hot_encode_categoricals)
            
        x = df.drop(columns='Target')
        y = df['Target']
            
        folds = split_data_k_folds(x, y, CrossValPipeline.n_folds)
            
        f1s_per_class = []
        avg_f1s = []
        best_iterations = []
        thresholds_all_folds = []
        
        for fold in folds:
            x_train, y_train, x_val, y_val = fold

            if scale_x:
                x_train, x_val = scale_data(x_train, x_val)

            model.fit(x_train, y_train,
                      eval_set=(x_val, y_val))

            if optimize_thresh:
                y_pred = model.predict(x_val, prediction_type='Probability')
                
                thresholds = threshold_optimizer.get_best_thresholds(model_package)
                thresholds_all_folds.append(thresholds)
                
                y_pred = make_preds_with_thresholds(y_pred, thresholds)
            else:
                y_pred = model.predict(x_val).flatten()

            f1_per_class = f1_score(y_val, y_pred, average=None)
            f1s_per_class.append(f1_per_class)
            avg_f1s.append(np.mean(f1_per_class))
            
            best_iterations.append(model.get_best_iteration())
        
        thresholds_all_folds = np.array(thresholds_all_folds)
        mean_per_class_thresholds = np.column_stack([thresholds_all_folds[j, :] for j in range(thresholds_all_folds.shape[1])]).tolist()
        
        print(thresholds_all_folds)
        print(mean_per_class_thresholds)
        
        return np.mean(avg_f1s)