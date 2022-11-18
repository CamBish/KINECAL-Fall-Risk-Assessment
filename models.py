import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

def train_xgboost(trainData, modelDir = "models/", modelName = "xgboost"):
    """Train a XGBoost model on provided data

    Args:
        data (pd.DataFrame): input data in the form of a pandas dataframe
    """
    X, y = trainData.iloc[:,:-1], trainData.iloc[:,-1]
    d = xgb.DMatrix(data=X, label=y)
        
    # Train model
    model = xgb.XGBClassifier()
    
    
    #save model
    model.save_model(modelDir + '/' + modelName + '.json')

    


svc = svm.SVC()



