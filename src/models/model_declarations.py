from imports import CatBoostClassifier

model_packages = [
        {'model' : CatBoostClassifier(iterations=100,
                        learning_rate=0.05,
                        early_stopping_rounds=10,
                        depth=6,
                        loss_function='MultiClass',
                        verbose=0,
                        random_seed=42,
                        thread_count=1
                        ),
        'using_test_set': False, 
        'one_hot_encode_categoricals': True, 
        'scale_x' : True, 
        'optimize_epochs' : True,
        'optimize_thresh' : False,
        'thresholds' : None,
        'best_epochs' : None
        },

        {'model' :CatBoostClassifier(iterations=100,
                        learning_rate=0.05,
                        early_stopping_rounds=10,             
                        depth=6,
                        loss_function='MultiClass',
                        verbose=0,
                        random_seed=42,
                        thread_count=1
                        ),
        
        'using_test_set': False, 
        'one_hot_encode_categoricals': True,
        'scale_x' : True, 
        'optimize_epochs' : False,
        'optimize_thresh' : False,
        'thresholds' : None,
        'best_epochs' : None
        }
]