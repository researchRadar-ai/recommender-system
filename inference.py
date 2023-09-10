import torch
import pandas as pd
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.utils.timer import Timer

from rating_model import Rating_Model
from dataset import Dataset as CustomDataset

# read the raw features
user = 1
paper = 1
Category = 1
Citation_Count = 1
Impact_Factor = 1
Time_Spent = 1
Number_of_Annotations = 1
input_data = [Category, Citation_Count, Impact_Factor, Time_Spent, Number_of_Annotations]


# mapping from raw feature to rating
input_size = 5 # Number of features (Category Citation_Count Impact_Factor Time_Spent Number_of_Annotations)
hidden_size = 8
rating_model = Rating_Model(input_size, hidden_size)

# rate = rating_model(input_data)
rating = int(sum(input_data) / len(input_data))

print("rating: ", rating)

# using the NCF model as recommender system

test = pd.DataFrame(columns=['userID', 'itemID', 'rating', 'timestamp'])
for i in range(2):
    test.loc[i] = [user, paper, rating, '2020-01-01']

train_file = "data/train.csv"
test_file = "data/inference.csv"
test.to_csv(test_file, index=False)

# Model parameters
EPOCHS = 50
BATCH_SIZE = 256

SEED = 42

data = CustomDataset(train_file=train_file, test_file=test_file, seed=SEED)

model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)

with Timer() as train_time: # have to perform training, otherwise, got "AttributeError: 'NCF' object has no attribute 'user2id'"
    model.fit(data)

# print("Took {} seconds for training.".format(train_time))


with Timer() as test_time:
    users, items, preds = [], [], []
    item = list(test.itemID.unique())
    for user in test.userID.unique():
        user = [user] * len(item) # for a given user, output the probability of recommending each item
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))
        print("preds: ", len(preds))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(test, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time))

