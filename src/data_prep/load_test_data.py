from imports import pd
from constants import DATA_PATH

def load_test_as_csv():
    test_df = pd.read_csv(DATA_PATH / 'Test.csv')
    print(test_df)