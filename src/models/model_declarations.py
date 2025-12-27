from imports import np, CatBoostClassifier
from constants import RANDOM_SEED, EARLY_STOPPING_ROUNDS

model_packages = [
        {'using_test_set': False, 
         'one_hot_encode_categoricals': True, 
         'scale_x' : True, 
         'optimize_epochs_and_lr' : True,
         'optimize_tree_depth' : True,
         'optimize_thresh' : True,
         'best_epochs' : None,
         'best_lr' : None,
         'best_tree_depth': None,
         'thresholds' : None,
         },
]

prediction_packages = [
        {'model' : CatBoostClassifier(iterations=261,
                                             learning_rate=np.float64(0.06842646032095057),
                                             early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                             depth=6,
                                             loss_function='MultiClass',
                                             verbose=0,
                                             random_seed=RANDOM_SEED,
                                             thread_count=1),
         'using_test_set': True, 
         'one_hot_encode_categoricals': True, 
         'scale_x' : True, 
         'thresholds' : [np.float64(0.32653061224489793), 
                         np.float64(0.44897959183673464), 
                         np.float64(0.26530612244897955)],
         },
        
]