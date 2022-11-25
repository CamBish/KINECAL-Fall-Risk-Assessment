import pandas as pd
import numpy as np
import os
from const import User, Exercise


def readKinecalFiles (exercise, user):
    """Reads and loads the input from the Kinecal dataset

    Args:
        exercise (string): The type of exercise to load data for
        user (User.user): User object to determine filepaths

    Returns:
        dataset_df: dataframe containing dataset values
    """   
    #Get filepath from user
    if(user == User.L): #Leonard is user
        dataDir = 'D:/kinecal-1.0.1/kinecal-1.0.1/'
    elif(user == User.CD): #Cam-DESKTOP is user
        dataDir = 'D:/kinecal-1.0.1/'
    elif(user == User.CL): #Cam-Laptop is user
        dataDir = 'D:/kinecal-1.0.1/'
    elif(user == User.CS): #Cam-Server is user
        dataDir = '' #not implemented!
    
    
    #assign filepaths for folders and input csv files
    folderDir = dataDir + '/kinecal/'
    csvFile = dataDir + 'register.csv'
    
    #obtain list of directories in folderDir
    folders = os.listdir(folderDir)
    
    #load register.csv into a dataframe
    subject_data_df = pd.read_csv(csvFile)
    subject_data_df = subject_data_df.loc[:,~subject_data_df.columns.str.match("Unnamed")]

    column_names = ['part_id', 'movement', 'group', 'sex', 'height', 'weight', 
                    'BMI', 'recorded_in_the_lab', 'clinically_at_risk', 'RDIST_ML',
                    'RDIST_AP', 'RDIST', 'MDIST_ML', 'MDIST_AP', 'MDIST', 'TOTEX_ML',
                    'TOTEX_AP', 'TOTEX', 'MVELO_ML', 'MVELO_AP', 'MVELO', 'MFREQ_ML',
                    'MFREQ_AP','MFREQ','AREA_CE']
    
    dataset_df = pd.DataFrame(columns=column_names)
    
    #iterate through each folder in folderDir
    for folder in folders:
        #find direcotry of each subject and specific exercise
        subjectDir = folderDir + folder + '/'
        exerciseDir = subjectDir + folder + '_' + Exercise.CHOICES[exercise] + '/'
        
        if (os.path.exists(exerciseDir)): #if exercise directory exists, read in data
            sway_metrics_df = pd.read_csv(exerciseDir + 'sway_metrics.csv')
            dataset_df = pd.concat([dataset_df,sway_metrics_df])
        
        else: #otherwise create dataframe with zero values
            zero_data = np.zeros(shape=(1,len(column_names)))
            empty_row_df = pd.DataFrame(zero_data, columns=column_names)
            empty_row_df['part_id'] = folder
            empty_row_df['movement'] = exercise
            
            participant_data = subject_data_df.loc[subject_data_df.part_id.isin(['SPPB'+folder])]
            attribute_list = ['group','sex','height','weight','BMI','recorded_in_the_lab']
            
            for attribute in attribute_list:
                empty_row_df[attribute] = participant_data[attribute].values
                
            empty_row_df['clinically_at_risk'] = participant_data['clinically-at-risk'].values
            dataset_df = pd.concat([dataset_df,empty_row_df])
    
    dataset_df = dataset_df.reset_index(drop=True)
    return dataset_df   

def replaceMissingValues (x_df, y_df):
    """Replaces missing values in the dataset with the mean of the class

    Args:
        x_df (pd.DataFrame): Sway metrics dataframe
        y_df (pd.DataFrame): Dataframe with subject metadata

    Returns:
        x_df: dataframe that has missing values replaced with the mean of the class
    """    
    #find columns with missing values and print
    missing_indexes = x_df[x_df.RDIST_ML == 0].index.values
    print(missing_indexes)
    #iterate through each column with missing values
    for index in missing_indexes:
        class_label = y_df.loc[index,'group']
        class_indexes = y_df.loc[y_df.group == class_label].index.values
        class_indexes = np.delete(class_indexes,np.where(class_indexes==index))
        class_df = x_df.loc[x_df.index.isin(class_indexes)]
        #replace missing values with mean of class
        for column in x_df:
            x_df.loc[index,column] = np.mean(class_df.loc[:,column].values)
    
    return x_df

def datasetNormalization (x_df):
    """ Normalizes each sample in the dataset i.e. (sample - sampleMean)/sampleVariance
    
    Args: 
        x_df (pd.Dataframe): Dataset Dataframe
    
    Returns:
        x_df: dataset with each sample noramlized
    """
    mean = np.asarray(x_df).mean(axis=1,keepdims=True)
    variance = np.asarray(x_df).var(axis=1,keepdims=True)
    x_df = x_df - mean
    x_df = x_df / variance
    return x_df
    