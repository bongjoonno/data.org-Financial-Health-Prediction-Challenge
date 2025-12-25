from imports import pd
from constants import DATA_PATH

def load_train_as_csv():
    train_df = pd.read_csv(DATA_PATH / 'Train.csv')
    print(train_df)