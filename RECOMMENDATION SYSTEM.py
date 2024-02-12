#Please Install the Surprise module. Thanks Ketan Kumawat.

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Sample data (user_id, item_id, rating)
data = [
    (1, 'Movie A', 4),
    (1, 'Movie B', 3),
    (1, 'Movie C', 5),
    (2, 'Movie A', 2),
    (2, 'Movie B', 4),
    (2, 'Movie C', 3),
    (3, 'Movie A', 5),
    (3, 'Movie B', 1),
    (3, 'Movie C', 4),
]

# Load data into Surprise dataset
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data, reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Build and train the collaborative filtering model (user-based)
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Get recommendations for a specific user
user_id = 1
user_items = set(dataset.df[dataset.df['userID'] == user_id]['itemID'])
unrated_items = set(dataset.df['itemID']) - user_items

predictions = [model.predict(user_id, item) for item in unrated_items]
recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

# Print recommendations
print(f"Recommendations for User {user_id}:")
for recommendation in recommendations:
    print(f"{recommendation.iid} - Estimated Rating: {recommendation.est:.2f}")
