import csv
import torch
import torch.nn as neural_network_module
import torch.optim as optimizer_module
import random

# Define the path to your CSV file
file_path = "1.csv"

# Function to read data from CSV file
def read_data(file_path):
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip the column headers

        data = []

        for row in reader:
            features = [float(x) for x in row[:-1]]
            target = float(row[-1])
            data.append((features, target))

    return data

# Load data
data = read_data(file_path)
print(data)

features = torch.tensor([d[0] for d in data])
targets = torch.tensor([d[1] for d in data])

# Create the network model
class Net(neural_network_module.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = neural_network_module.Linear(len(data[0][0]), 128)
        self.fc2 = neural_network_module.Linear(128, 64)
        self.fc3 = neural_network_module.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # Create an instance of the network
model = Net()

# Define optimizer and loss function
optimizer = optimizer_module.Adam(model.parameters())
criterion = neural_network_module.MSELoss()

num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    epoch += 1

    # Shuffle data
    shuffled_data = list(zip(features, targets))
    random.shuffle(shuffled_data)
    features, targets = zip(*shuffled_data)

    # Loop through each data point
    for x, y in zip(features, targets):
        # Forward pass
        prediction = model(x.view(1, -1))

        # Calculate loss
        loss = criterion(prediction, y.view(1, -1))

        # Backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every epoch
    if epoch % 25 == 0:
        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

agents = [
    'Brimstone',
    'Viper',
    'Omen',
    'Killjoy',
    'Cypher',
    'Sova',
    'Sage',
    'Phoenix',
    'Jett',
    'Reyna',
    'Raze',
    'Breach',
    'Skye',
    'Yoru',
    'Astra',
    'KAY/O',
    'Chamber',
    'Neon',
    'Fade',
    'Harbor',
    'Gekko',
    'Deadlock',
    'Iso',
]
maps = [
    'Ascent',
    'Bind',
    'Breeze',
    'Fracture',
    'Haven',
    'Icebox',
    'Lotus',
    'Pearl',
    'Split',
    'Sunset',
]
ranks = [
    'Iron 1',
    'Iron 2',
    'Iron 3',
    'Bronze 1',
    'Bronze 2',
    'Bronze 3',
    'Silver 1',
    'Silver 2',
    'Silver 3',
    'Gold 1',
    'Gold 2',
    'Gold 3',
    'Platinum 1',
    'Platinum 2',
    'Platinum 3',
    'Diamond 1',
    'Diamond 2',
    'Diamond 3',
    'Ascendant 1',
    'Ascendant 2',
    'Ascendant 3',
    'Immortal 1',
    'Immortal 2',
    'Immortal 3',
    'Radiant',
]
match_results = [
    'Loss',
    'Win',
    'Draw',
]

# Predict on a new data point
def predict(inputs):
    inputs[0] = ranks.index(inputs[0])
    inputs[1] = maps.index(inputs[1])
    inputs[2:7] = [agents.index(agent) for agent in inputs[2:7]]

    inputs = torch.tensor(inputs, dtype = torch.float32)

    prediction = model(inputs)
    prediction = prediction.item()
    prediction = 0 if prediction < 0 else prediction

    winrate = str(round(prediction * 100)) + '%'

    print(f"Calculated Winrate: {winrate}")