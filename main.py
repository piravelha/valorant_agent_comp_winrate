# Import necessary libraries
import torch

# Load the model
model = torch.load("model.pth")

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

def preprocess_data(data):
    # Preprocess the data (replace this with your specific preprocessing steps)
    processed_data = ranks.index(processed_data[0])
    processed_data[1] = maps.index(processed_data[1])
    processed_data[2:7] = [agents.index(agent) for agent in processed_data[2:7]]

    inputs = torch.tensor(processed_data, dtype = torch.float32)

    return processed_data

# Define your prediction function
def make_prediction(data):
    # Preprocess the data (replace this with your specific preprocessing steps)
    processed_data = preprocess_data(data)

    # Feed the data to the model
    output = model(processed_data)

    # Post-process the output (replace this with your specific post-processing steps)
    prediction = model(data)
    prediction = prediction.item()
    prediction = 0 if prediction < 0 else prediction

    winrate = str(round(prediction * 100)) + '%'

    print(f"Calculated Winrate: {winrate}")

    return winrate

# Example usage
#data = ...  # your input data
#prediction = predict(data)

#print(f"Prediction: {prediction}")

import gradio as gr

# Create Gradio interface with relevant inputs
interface = gr.Interface(
    fn=make_prediction,
    inputs=[
        # Input for rank
        gr.Dropdown(label="Rank", choices=ranks, default=ranks[0]),
        # Input for map
        gr.Dropdown(label="Map", choices=maps, default=maps[0]),
        # Input for agents
        gr.MultiSelect(label="Agent Picks (1-5)", choices=agents, default=[agents[0]]),
    ],
    outputs="text",
)

interface.launch()
