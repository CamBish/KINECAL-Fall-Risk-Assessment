import pandas as pd
import numpy as np
import os

def readKinecalFiles (excercise):
    folder_dir = 'D:/kinecal-1.0.1/kinecal-1.0.1/kinecal/'
    folders = os.listdir(folder_dir)

    subject_data_df = pd.read_csv('D:/kinecal-1.0.1/kinecal-1.0.1/register.csv')
    subject_data_df = subject_data_df.loc[:,~subject_data_df.columns.str.match("Unnamed")]

    column_names = ['part_id','movement','group','sex','height','weight','BMI','recorded_in_the_lab','clinically_at_risk','RDIST_ML','RDIST_AP','RDIST','MDIST_ML','MDIST_AP','MDIST','TOTEX_ML','TOTEX_AP','TOTEX','MVELO_ML','MVELO_AP','MVELO','MFREQ_ML','MFREQ_AP','MFREQ','AREA_CE']
    dataset_df = pd.DataFrame(columns=column_names)
    i=0
    for folder in folders:
        subject_dir = folder_dir+folder+'/'
        excercise_dir = subject_dir+folder+'_'+excercise+'/'
        if (os.path.exists(excercise_dir)):
            sway_metrics_df = pd.read_csv(excercise_dir + 'sway_metrics.csv')
            dataset_df = pd.concat([dataset_df,sway_metrics_df])
        else:
            zero_data = np.zeros(shape=(1,len(column_names)))
            empty_row_df = pd.DataFrame(zero_data, columns=column_names)
            empty_row_df['part_id'] = folder
            empty_row_df['movement'] = excercise
            participant_data = subject_data_df.loc[subject_data_df.part_id.isin(['SPPB'+folder])]
            attribute_list = ['group','sex','height','weight','BMI','recorded_in_the_lab']
            for attribute in attribute_list:
                empty_row_df[attribute] = participant_data[attribute].values
            empty_row_df['clinically_at_risk'] = participant_data['clinically-at-risk'].values
            dataset_df = pd.concat([dataset_df,empty_row_df])
    dataset_df = dataset_df.reset_index(drop=True)
    return dataset_df   

def replaceMissingValues (x_df, y_df):
    missing_indexes = x_df[x_df.RDIST_ML == 0].index.values
    print(missing_indexes)
    for index in missing_indexes:
        class_label = y_df.loc[index,'group']
        class_indexes = y_df.loc[y_df.group == class_label].index.values
        class_indexes = np.delete(class_indexes,np.where(class_indexes==index))
        class_df = x_df.loc[x_df.index.isin(class_indexes)]
        for column in x_df:
            x_df.loc[index,column] = np.mean(class_df.loc[:,column].values)
    
    #print(x_df)
    return x_df