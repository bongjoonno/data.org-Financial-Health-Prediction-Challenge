from imports import np, f1_score

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