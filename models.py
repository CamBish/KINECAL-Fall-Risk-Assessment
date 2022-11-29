import xgboost as xgb
import tensorflow as tf

import keras.layers.activation
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
import keras

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

    #Optional Additional parameters
    #   eval_set=[(X_train, y_train), (X_vali, y_vali)], 
    #   eval_metric=eval, 
    #   verbose=True
    
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

def train_model_callbacks(model: keras.models.Model, X, y, 
                          optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(), 
                          lr: float = 0.5, loss: str = "binary_crossentropy", metrics: list = None, 
                          epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2, 
                          verbose: int = 1, monitor: str = 'val_loss') -> keras.models.Model:
    
    if metrics is None:
        metrics = ["accuracy"]
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor=monitor
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=20, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(optimizer=optimizer(learning_rate=lr), loss=loss, metrics=metrics)

    model.fit(X, y, epochs=epochs, 
                batch_size=batch_size, 
                callbacks=callbacks,
                validation_split=validation_split,
                verbose=verbose
            )
    