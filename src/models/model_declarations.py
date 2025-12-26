from imports import CatBoostClassifier

model_packages = [
        {'using_test_set': False, 
         'one_hot_encode_categoricals': True, 
         'scale_x' : True, 
         'optimize_epochs_and_lr' : True,
         'optimize_thresh' : False,
         'thresholds' : None,
         'best_epochs' : None,
         'best_lr' : None
         },
        
        {'using_test_set': False, 
         'one_hot_encode_categoricals': True,
         'scale_x' : True, 
         'optimize_epochs_and_lr' : False,
         'optimize_thresh' : False,
         'thresholds' : None,
         'best_epochs' : None,
         'best_lr' : None
        }
]