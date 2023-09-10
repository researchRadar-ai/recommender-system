'''
    This file contains Python script to create a dummy dataset for our recommender system, just to get things started.
'''

import random
import pandas as pd


'''
    ProjectID PaperID Rating(1-10) Category Citation_Count Impact_Factor Time_Spent Number_of_Annotations
'''
def build_dataset_full(size):

    dataset = pd.DataFrame(columns=['ProjectID', 'PaperID', 'Rating', 'Category', 'Title', 'Citation_Count', 'Impact_Factor', 'Time_Spent', 'Number_of_Annotations'])
    for i in range(size):
        dataset.loc[i] = [random.randint(1, 100), # ProjectID 
                          random.randint(1, 1000), # PaperID
                          random.randint(1, 10), # Rating
                          random.randint(1, 20), # Category
                          random.randint(10, 200), # Citation_Count
                          random.randint(1, 15), # Impact_Factor
                          random.randint(1, 100), # Time_Spent 
                          random.randint(1, 100) # Number_of_Annotations
                          ]
    
    dataset.to_csv('dataset.csv', index=False)


'''
    userID itemID rating timestamp
'''
def build_dataset(size):

    dataset = pd.DataFrame(columns=['userID', 'itemID', 'rating', 'timestamp'])
    for i in range(size):
        dataset.loc[i] = [random.randint(1, 100), # userID 
                          random.randint(1, 1000), # itemID
                          random.randint(1, 10), # rating
                          f'{random.randint(2015, 2020)}-0{random.randint(1, 9)}-0{random.randint(1, 25)}' # timestamp
                          ]
    
    dataset.to_csv('data/dataset.csv', index=False)



if __name__ == '__main__':
    
    build_dataset(1000)

