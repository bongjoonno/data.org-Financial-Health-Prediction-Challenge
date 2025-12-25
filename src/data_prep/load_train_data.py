from imports import pd
from constants import DATA_PATH

train_df = pd.read_csv(DATA_PATH / 'Train.csv')
train_df = train_df.drop(columns='ID')