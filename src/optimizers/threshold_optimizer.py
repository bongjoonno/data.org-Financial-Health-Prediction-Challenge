from imports import np, f1_score

class ThresholdOptimizer:
    n_folds = 5
    
    @staticmethod
    def get_best_thresholds(model):
        x, y = make_x_and_y(df, clean_data_args)
        
        folds = split_k_folds(x, y, ThresholdOptimizer.n_folds)

        avg_threshes = [[], [], []]
        for fold in folds:
            x_train, y_train, x_val, y_val = fold

            if scale_x:
                x_train, x_val = scale_data(x_train, x_val, columns_to_scale=cols_to_scale)

            model.fit(x_train, y_train)

   
            y_pred = model.predict(x_val, prediction_type='Probability')

            optim = ThresholdOptimizer.optimize_threshold(y_pred, y_val, 3)
            thresholds = []

            for k, v in optim.items():
                max_f1_thresh_pair = sorted(v.items(), key=lambda pair: pair[1], reverse=True)[0]
                max_f1_thresh = max_f1_thresh_pair[0]

                avg_threshes[k] = max_f1_thresh

                thresholds.append(max_f1_thresh)

            y_pred = make_preds(y_pred, thresholds)



        return [np.mean(threshes) for threshes in avg_threshes]

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