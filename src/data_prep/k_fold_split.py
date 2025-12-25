from imports import pd

def split_data_k_folds(x, y, k):
    folds = []

    test_size = int(len(x)*(1/k))

    for i in range(0, len(x)-test_size, test_size):
        x_train = pd.concat([x[:i], x[i+test_size:]])
        x_val = x[i: i+test_size]

        y_train = pd.concat([y[:i], y[i+test_size:]])
        y_val = y[i: i+test_size]

        folds.append((x_train, y_train, x_val, y_val))

    return folds