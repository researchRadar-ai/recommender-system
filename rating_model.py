import torch
import torch.nn as nn

# Define the architecture of the neural network
class Rating_Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Rating_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, 1)  # Hidden layer to output layer with 1 neuron

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = 9 * torch.sigmoid(out) + 1  # Sigmoid output scaled to (1, 10) range
        return int(out)

# Define the input size and hidden size
input_size = 5 # Number of features (Category Citation_Count Impact_Factor Time_Spent Number_of_Annotations)
hidden_size = 8

# # Create an instance of the FeedForwardNN model
# model = Rating_Model(input_size, hidden_size)

# Category = 1
# Citation_Count = 1.0
# Impact_Factor = 1
# Time_Spent = 1
# Number_of_Annotations = 1
# input_data = torch.tensor([Category, Citation_Count, Impact_Factor, Time_Spent, Number_of_Annotations])


# # Example usage:
# output = model(input_data)

# print("Output:", output)

