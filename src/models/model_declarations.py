from imports import CatBoostClassifier

def load_all_models():
    model_packages = [
            {'model' : CatBoostClassifier(iterations=100,
                                learning_rate=0.1,
                                depth=6,
                                loss_function='MultiClass',
                                verbose=False,
                                random_seed=42,
                                thread_count=1
                                ),
            'using_test_set': False, 
            'one_hot_encode_categoricals': True, 
            'scale_x' : True, 
            'optimize_thresh' : True,
            'thresholds' : None,
            },

            {'model' :CatBoostClassifier(iterations=100,
                                learning_rate=0.1,
                                depth=6,
                                loss_function='MultiClass',
                                verbose=False,
                                random_seed=42,
                                thread_count=1
                                ),
            
            'using_test_set': False, 
            'one_hot_encode_categoricals': True,
            'scale_x' : True, 
            'optimize_thresh' : False,
            'thresholds' : None
            }
    ]
    
    return model_packages     