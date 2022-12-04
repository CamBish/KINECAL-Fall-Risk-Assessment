import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from multiprocessing import cpu_count


def train_xgboost(X, y, param, eval= "logloss") -> xgb.XGBClassifier:
    """Train a XGBoost model on provided data using sklearn API.

    Args:
        data (pd.DataFrame): input data in the form of a pandas dataframe
        param (dict): dictionary of parameters for XGBoost
        eval (str): evaluation metric for XGBoost
    
    Returns:
        xgb.XGBClassifier: trained XGBoost model
    """
    
    #create XGBoost model from parameters
    model = xgb.XGBClassifier(**param, tree_method="gpu_hist", validate_parameters=True, n_jobs=cpu_count())
    
    print("Training XGBoost model...")
    
    #train model
    model.fit(X, y, eval_metric=eval)
    
    return model


def tune_xgboost(X, y, param_grid: dict, cv: int = 2, verbose: int = 1) -> xgb.XGBClassifier:
    """Tune XGBoost model hyperparameters using GridSearchCV and sklearn API.

    Args:
        X (any): input data
        y (any): data labels
        param_grid (dict): dictionary of parameters to tune for XGBoost using GridSearchCV
        cv (int, optional): number of cross-validation folds. Defaults to 2.
        modelDir (str, optional): output directory for models. Defaults to "models/".
        modelName (str, optional): name of fine-tuned model. Defaults to "tunedXGBoost".

    Returns:
        str: path to the saved model
    """
    #create XGBoost model for tuning
    xgb_model = xgb.XGBClassifier(tree_method="gpu_hist",validate_parameters=True, n_jobs=cpu_count())

    grid = GridSearchCV(xgb_model, param_grid=param_grid, cv=cv, verbose=verbose, return_train_score=True)
    grid.fit(X, y)

    # Print the best parameters and score
    print(f"Best parameters: {str(grid.best_params_)}")
    print(f"Best score: {str(grid.best_score_)}")

    return grid.best_estimator_
    

def train_svm(X, y, params: dict) -> svm.SVC:
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        params (dict): _description_
        modelDir (str, optional): _description_. Defaults to "models/".
        modelName (str, optional): _description_. Defaults to "svm".

    Returns:
        svc: trained SVM model for classification
    """       
    svc = svm.SVC(**params)
    
    print("Training SVM model...")
    
    svc.fit(X, y)
    
    return svc


def tune_svm(X, y, param_grid: dict, cv: int = 2) -> svm.SVC:
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

    grid = GridSearchCV(svm_model, param_grid=param_grid, cv=cv, verbose=1, return_train_score=True)
    grid.fit(X, y)

    # Print the best parameters and score
    print(f"Best parameters: {str(grid.best_params_)}")
    print(f"Best score: {str(grid.best_score_)}")

    return grid.best_estimator_
