import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from multiprocessing import cpu_count
    

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
