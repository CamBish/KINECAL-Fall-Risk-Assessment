import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import f1_score, roc_auc_score
import time
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import numpy as np

def tune_svm(X, y,  param_grid: dict, isMulticlass: bool = False, cv: int = 2) -> svm.SVC:
    """Tune SVM model hyperparameters using GridSearchCV and sklearn API.

    Args:
        X (any): input data
        y (any): data labels
        param_grid (dict): dictionary of parameters to tune for SVM using GridSearchCV
        cv (int, optional): number of cross-validation folds. Defaults to 2.

    Returns:
        best_model: best performing SVM classifier model from GridSearchCV
    """  
    #create SVM model for tuning
    svm_model = svm.SVC()
    if isMulticlass:
        grid = GridSearchCV(svm_model, scoring='f1_weighted',param_grid=param_grid, cv=cv, verbose=1, return_train_score=True)
    else:
        grid = GridSearchCV(svm_model, scoring='roc_auc', param_grid=param_grid, cv=cv, verbose=1, return_train_score=True)
    
    grid.fit(X, y)

    # Print the best parameters and score
    print(f"Best parameters: {str(grid.best_params_)}")
    print(f"Best score: {str(grid.best_score_)}")

    return grid.best_estimator_


def hyperopt_multiclass(param_space: any, xTrain: any, yTrain: any, xTest: any, yTest: any, num_rounds: int = 100):
    """Hyperparameter tuning using Bayesian Optimization for a multiclass XGBoost classifier using hyperopt library.

    Args:
        param_space (any): hyperparameter space to tune
        xTrain (any): training data
        yTrain (any): training labels
        xTest (any): test data
        yTest (any): test labels
        num_rounds (int, optional): maximum number of parameter combination rounds. Defaults to 100.

    Returns:
        xgb.XGBClassifier, dict: best performing XGBoost classifier model and best parameters
    """    
    start = time.time()
    
    def objective(param_space):
        clf = xgb.XGBClassifier(**param_space)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(clf, xTrain, yTrain, cv=skf, scoring='f1_weighted').mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best_param = fmin(fn=objective, 
                    space=param_space, 
                    algo=tpe.suggest, 
                    max_evals=num_rounds,
                    trials=trials)
    loss = [x['result']['loss'] for x in trials.trials]
    
    #ensure best_param dictionary has right data types
    best_param['max_depth'] = int(best_param['max_depth'])
    best_param['tree_method'] = ['exact', 'gpu_hist'][best_param['tree_method']]
    best_param['objective'] = ['multi:softmax', 'multi:softprob'][best_param['objective']]
    
    best_clf = xgb.XGBClassifier(**best_param)
    best_clf.fit(xTrain, yTrain)
    
    print('==============RESULTS==============')
    print('Best parameters: ', best_param)
    print('Best loss: ', -np.min(loss))
    print('Time taken: ', time.time() - start)
    print('Test accuracy: ', f1_score(yTest, best_clf.predict(xTest), average='weighted'))
    print('Parameter combinations evaluated: ', len(trials.trials))
    return best_clf, best_param


def hyperopt_binary(param_space: any, xTrain: any, yTrain: any, xTest: any, yTest: any, num_rounds: int = 100):
    """Hyperparameter tuning using Bayesian Optimization for a binary XGBoost classifier using hyperopt library.

    Args:
        param_space (any): hyperparameter space to tune
        xTrain (any): training data
        yTrain (any): training labels
        xTest (any): test data
        yTest (any): test labels
        num_rounds (int, optional): maximum number of parameter combination rounds. Defaults to 100.

    Returns:
        xgb.XGBClassifier, dict: best performing XGBoost classifier model and best parameters
    """    
    start = time.time()
    
    def objective(param_space):
        clf = xgb.XGBClassifier(**param_space)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(clf, xTrain, yTrain, cv=skf, scoring='roc_auc').mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    trials = Trials()
    best_param = fmin(fn=objective, 
                    space=param_space, 
                    algo=tpe.suggest, 
                    max_evals=num_rounds,
                    trials=trials)
    loss = [x['result']['loss'] for x in trials.trials]
    
    #ensure best_param dictionary has right data types
    best_param['max_depth'] = int(best_param['max_depth'])
    best_param['tree_method'] = ['exact', 'gpu_hist'][best_param['tree_method']]
    best_param['objective'] = ['binary:logistic', 'binary:logitraw', 'binary:hinge'][best_param['objective']]
    
    best_clf = xgb.XGBClassifier(**best_param)
    best_clf.fit(xTrain, yTrain)
    
    print('==============RESULTS==============')
    print('Best parameters: ', best_param)
    print('Best loss: ', -np.min(loss))
    print('Time taken: ', time.time() - start)
    print('Test accuracy: ', roc_auc_score(yTest, best_clf.predict(xTest), average='weighted'))
    print('Parameter combinations evaluated: ', len(trials.trials))
    return best_clf, best_param