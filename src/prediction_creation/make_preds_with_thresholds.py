
def make_preds_with_thresholds(probs_list, thresholds):
  preds = []

  for probs in probs_list:
    cur_pred = [0, 0, 0]

    for i, prob in enumerate(probs):
      threshold = thresholds[i]
      if prob >= threshold:
        cur_pred[i] = prob

    preds.append(cur_pred.index(max(cur_pred)))

  return preds