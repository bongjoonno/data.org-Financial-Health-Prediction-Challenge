from imports import np, f1_score
from src.data_prep import clean_data, train_df, split_data_k_folds, scale_data
from src.prediction_creation import make_preds_with_thresholds

class HyperParamOptimizer:
    n_folds = 5
    
    @staticmethod
    def get_best_thresholds(model_package: dict):
        model = model_package['model']
        one_hot_encode_categoricals = model_package['one_hot_encode_categoricals']
        scale_x = model_package['scale_x']
        
        
        df = clean_data(train_df, one_hot_encode_categoricals)
        
        x = df.drop(columns='Target')
        y = df['Target']
        
        folds = split_data_k_folds(x, y, HyperParamOptimizer.n_folds)

        avg_threshes = [[], [], []]
        
        for fold in folds:
            x_train, y_train, x_val, y_val = fold

            if scale_x:
                x_train, x_val = scale_data(x_train, x_val)

            model.fit(x_train, y_train)

   
            y_pred = model.predict(x_val, prediction_type='Probability')

            classes_thresh_dict = HyperParamOptimizer.optimize_threshold(y_pred, y_val, 3)
            thresholds = []

            for k, v in classes_thresh_dict.items():
                max_f1_thresh_pair = sorted(v.items(), key=lambda pair: pair[1], reverse=True)[0]
                max_f1_thresh = max_f1_thresh_pair[0]

                avg_threshes[k] = max_f1_thresh

                thresholds.append(max_f1_thresh)

            y_pred = make_preds_with_thresholds(y_pred, thresholds)

        model_package['thresholds'] = [np.mean(threshes) for threshes in avg_threshes]

    @staticmethod
    def optimize_threshold(probs_list, y_true, num_classes):
        probs_per_class = [[] for _ in range(num_classes)]
  
        thresholds = np.linspace(0, 1, 50)

        for probs in probs_list:
            for i, prob in enumerate(probs):
                probs_per_class[i].append(prob)
    
        classes_thresh_dict = {c: {} for c in range(num_classes)}

        for i, classs in enumerate(probs_per_class):
            y_true_binary = [1 if y==i else 0 for y in y_true]
            
            for threshold in thresholds:
                preds = [1 if prob >= threshold else 0 for prob in classs]
                f1 = f1_score(y_true_binary, preds, average='binary')
                classes_thresh_dict[i][threshold] = f1

        return classes_thresh_dict

    @staticmethod
    def optimize_epochs_and_learning_rate(model_package: dict):
        model = model_package['model']
        one_hot_encode_categoricals = model_package['one_hot_encode_categoricals']
        scale_x = model_package['scale_x']
    
        df = clean_data(train_df, one_hot_encode_categoricals)
                
        x = df.drop(columns='Target')
        y = df['Target']
            
        folds = split_data_k_folds(x, y, HyperParamOptimizer.n_folds)
        
        epochs_learning_rate_pairs: dict[tuple, float] = {}
        
        learning_rates = np.random.uniform(0.00001, 0.1, 10)
        
        for lr in learning_rates:
            for fold in folds:
                x_train, y_train, x_val, y_val = fold

                if scale_x:
                    x_train, x_val = scale_data(x_train, x_val)

                model.fit(x_train, y_train,
                          learning_rate = lr,
                          eval_set=(x_val, y_val))

                best_iteration = model.get_best_iteration()
            
                model.fit(x_train, y_train, 
                          iterations = best_iteration,
                          learning_rate = lr)
                
                y_pred = model.predict(x_val, y_val).flatten()
                macro_f1 = f1_score(y_val, y_pred, average='Macro')
                
                epochs_learning_rate_pairs[(best_iteration, lr)] = macro_f1
        
        for pair, f1 in epochs_learning_rate_pairs.items():
            print(pair, f1)