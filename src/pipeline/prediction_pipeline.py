from imports import np, pd, Path
from constants import DEFAULT_THRESHOLDS
from src.data_prep import clean_data, scale_data, train_df, test_df
from src.prediction_creation import make_preds_with_thresholds

test_df_no_id = test_df.drop(columns='ID')
preds_path = Path(r'D:\code\repos\data.org-Financial-Health-Prediction-Challenge\preds')

class PredictionPipeline:
    @staticmethod
    def make_testing_preds(model_package: dict):
        preds = PredictionPipeline.train_and_make_preds(model_package)
        preds_file_name = input('enter file name: ')
        
        preds.to_csv(preds_path / preds_file_name / '.csv', index=False)
        
    @staticmethod   
    def train_and_make_preds(model_package: dict):
        model = model_package['model']
        using_test_set = model_package['using_test_set']
        one_hot_encode_categoricals = model_package['one_hot_encode_categoricals']
        scale_x = model_package['scale_x']
        thresholds = model_package.get('thresholds', DEFAULT_THRESHOLDS)
        
        df_train = clean_data(train_df, one_hot_encode_categoricals)
        
        x_train = df_train.drop(columns='Target')
        y_train = df_train['Target']
        
        x_test = clean_data(test_df_no_id, one_hot_encode_categoricals, using_test_set)
        
        if scale_x:
            x_train, x_test = scale_data(x_train, x_test)
           
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test, prediction_type='Probability')

        y_pred = pd.Series(make_preds_with_thresholds(y_pred, thresholds)) 
        
        results = pd.concat([test_df['ID'], y_pred], ignore_index=True)
        return results