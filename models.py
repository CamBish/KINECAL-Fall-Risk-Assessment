import xgboost as xgb
import tensorflow as tf

import keras.layers.activation
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
import keras



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from multiprocessing import cpu_count
import os
from pickle import dump

def train_xgboost(X, y, param, eval= "logloss", modelDir = "models/", modelName = "xgboost"):
    """Train a XGBoost model on provided data using sklearn API.

    Args:
        data (pd.DataFrame): input data in the form of a pandas dataframe
        param (dict): dictionary of parameters for XGBoost
        eval (str): evaluation metric for XGBoost
        modelDir (str): directory to save the model
        modelName (str): name of the model for saving
    
    Returns:
        str: path to the saved model
    """
    #check if model directory exists, if not create it
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
        
    #create output path
    outPath = modelDir + modelName + '.json'
    
    #split data into train and test sets
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #create XGBoost model from parameters
    model = xgb.XGBClassifier(**param, tree_method="gpu_hist", validate_parameters=True, n_jobs=cpu_count())
    
    print("Training XGBoost model...")
    
    #train model
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_vali, y_vali)], 
              eval_metric=eval, 
              verbose=True)
    
    #save model
    model.save_model(outPath)
    
    print("Model saved to: " + outPath)
    return outPath


def tune_xgboost(X, y, param_grid: dict, cv: int = 2, modelDir: str = "models/", modelName: str = "tunedXGBoost") -> str:
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
    #check if model directory exists, if not create it
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    
    #create output path
    outPath = modelDir + modelName + '.json'
    
    #create XGBoost model for tuning
    xgb_model = xgb.XGBClassifier(tree_method="gpu_hist",validate_parameters=True, n_jobs=cpu_count())
    
    grid = GridSearchCV(xgb_model, param_grid=param_grid, cv=cv, verbose=1, return_train_score=True)
    grid.fit(X, y)
    
    # Print the best parameters and score
    print("Best parameters: " + str(grid.best_params_))
    print("Best score: " + str(grid.best_score_))
    
    # Save the best model
    best_model = grid.best_estimator_
    print(outPath)
    best_model.save_model(outPath)
    
    return outPath
    

def train_svm(X, y, params: dict, modelDir: str = "models/", modelName: str = "svm"):
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
    #check if model directory exists, if not create it
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    
    #split data into train and test sets
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svc = svm.SVC(**params)
    
    print("Training SVM model...")
    
    svc.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_vali, y_vali)], 
              eval_metric=eval, 
              verbose=True)
    
    return svc


def tune_svm(X, y, param_grid: dict, cv: int = 2, modelDir: str = "models/", modelName: str = "tunedSVM") -> str:
    """Tune SVM model hyperparameters using GridSearchCV and sklearn API.

    Args:
        X (any): input data
        y (any): data labels
        param_grid (dict): dictionary of parameters to tune for SVM using GridSearchCV
        cv (int, optional): number of cross-validation folds. Defaults to 2.
        modelDir (str, optional): output directory for models. Defaults to "models/".
        modelName (str, optional): name of fine-tuned model. Defaults to "tunedSVM".

    Returns:
        best_model: best performing SVM classifier model from GridSearchCV
    """  
    #check if model directory exists, if not create it
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    
    #create output path
    outPath = modelDir + modelName + '.pkl'
    
    #create SVM model for tuning
    svm_model = svm.SVC()
    
    grid = GridSearchCV(svm_model, param_grid=param_grid, cv=cv, verbose=1, return_train_score=True)
    grid.fit(X, y)
    
    # Print the best parameters and score
    print("Best parameters: " + str(grid.best_params_))
    print("Best score: " + str(grid.best_score_))
    
    # Save the best model
    best_model = grid.best_estimator_
    
    return best_model

def generate_deep_model(input_shape: int, output_shape: int, 
                        model_width:int = 128, hid_act_fxn: str = "relu", 
                        final_act_fxn: str = "sigmoid", summary: bool = True) -> tf.keras.Model:
    """Generate deep neural network model for classification.

    Args:
        input_shape (int): number of features in input data
        output_shape (int): number of classes in output data
        model_width (int, optional): Model width parameter. Defaults to 128.
        inner_act_fxn (str, optional): Activation function for hidden layers. Defaults to "relu".
        final_act_fxn (str, optional): Activation function for output layer. Defaults to "sigmoid".
        summary (bool, optional): Turn on model summary. Defaults to True.

    Returns:
        tf.keras.Model: deep neural network model for classification
    """       
    
    ip = Input(shape=input_shape)
    
    x = Dense(model_width)(ip)
    
    x = Activation(hid_act_fxn)(x)
    
    x = Dense(model_width)(x)
    
    x = Activation(hid_act_fxn)(x)
    
    x = Dense(model_width)(x)
    
    x = Activation(hid_act_fxn)(x)
    
    out = Dense(output_shape, activation=final_act_fxn)(x)
    
    model = Model(ip, out)
    
    if summary:
        model.summary()
    
    return model

def train_model(model: keras.models.Model, x_train, y_train, optimizer: str = "adam", 
                loss: str = "binary_crossentropy", metrics: list = ["accuracy"], 
                epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2, 
                verbose: int = 1) -> keras.models.Model:
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)