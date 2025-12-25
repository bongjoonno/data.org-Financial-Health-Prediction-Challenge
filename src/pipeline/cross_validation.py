def train_cross_val(model, clean_data_args: dict, scale_x: bool, optimize_thresh: bool):
  x, y = make_x_and_y(df, clean_data_args)

  f1s_per_class = []
  avg_f1s = []

  n_folds = 5
  folds = split_k_folds(x, y, n_folds)

  avg_threshes = [[], [], []]
  for fold in folds:
    x_train, y_train, x_val, y_val = fold

    if scale_x:
      x_train, x_val = scale_data(x_train, x_val, columns_to_scale=cols_to_scale)

    model.fit(x_train, y_train)

    if optimize_thresh:
      y_pred = model.predict(x_val, prediction_type='Probability')

      optim = optimize_threshold(y_pred, y_val, 3)
      thresholds = []

      for k, v in optim.items():
        max_f1_thresh_pair = sorted(v.items(), key=lambda pair: pair[1], reverse=True)[0]
        max_f1_thresh = max_f1_thresh_pair[0]

        avg_threshes[k] = max_f1_thresh

        thresholds.append(max_f1_thresh)

      y_pred = make_preds(y_pred, thresholds)
    else:
      y_pred = model.predict(x_val)

    f1_per_class = f1_score(y_val, y_pred, average=None)
    f1s_per_class.append(f1_per_class)
    avg_f1s.append(np.mean(f1_per_class))

  print(np.mean(avg_f1s))
  if optimize_thresh:
    return [np.mean(threshes) for threshes in avg_threshes]