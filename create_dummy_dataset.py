'''
    This file contains Python script to create a dummy dataset for our recommender system, just to get things started.
'''

# To mimic the MIND dataset (https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
# we have the following files
# behaviors.tsv - The click histories and impression logs of users
# papers.tsv - The information of research papers
# entity_embedding.vec (directly taken from MIND)
# relation_embedding.vec (directly taken from MIND)


import random
import pandas as pd


'''
    Paper ID:
    Category:
    Title:
    Citation Count:
    Journal/Conference Impact Factor:
    User Time Spent:
    Number of Annotations:

    Return: papers.tsv
'''
def build_paper(num_paper,
                num_category=10
                ):
    
    paper_df = pd.DataFrame(columns=['Paper ID', 'Category', 'Title', 'Citation Count', 'Impact Factor', 'User Time Spent', 'Number of Annotations'])

    for i in range(num_paper):
        paper_df.loc[i] = [i, random.randint(0, num_category), 'paper_' + str(i), random.randint(10, 200), random.randint(1, 15), random.randint(0, 100), random.randint(0, 100)]

    paper_df.to_csv('papers.tsv', sep='\t', index=False)


'''
    Impression ID: The ID of an impression
    User ID: The anonymous ID of a user
    Time: The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM"
    History: The paper click history (ID list of clicked papers) of this user before this impression. The clicked research papers are ordered by time
    Impressions: List of papers displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled

    Return: behaviors.tsv    
'''
def build_behavior(num_user):

    behavior_df = pd.DataFrame(columns=['Impression ID', 'User ID', 'Time', 'History', 'Impressions'])

    for i in range(num_user):
        behavior_df.loc[i] = [random.randint(0, 100), 
                              'U' + str(i), 
                              f'{random.randint(1, 12)}/{random.randint(1, 30)}/{random.randint(2010, 2023)} {random.randint(0, 11)}:{random.randint(1, 59)}:{random.randint(1, 59)} {random.choice(["AM", "PM"])}', 
                              'N' + str(random.randint(100000, 999999)), 
                              'N' + str(random.randint(100000, 999999)) + random.choice(['-0', '-1'])]

    behavior_df.to_csv('behaviors.tsv', sep='\t', index=False)


if __name__ == '__main__':
    
    num_paper = 1000
    num_user = 100

    build_paper(num_paper)
    build_behavior(num_user)







