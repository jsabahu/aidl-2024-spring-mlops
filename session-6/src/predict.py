from typing import List

import torch
import torch.nn.functional as f

from model import RegressionModel


@torch.no_grad()
def predict(input_features: List[float]):
    # load the checkpoint from the correct path
    modelpath = "/checkpoints/checkpoint.pt"
    checkpoint = torch.load(modelpath)

    x_mean = checkpoint['x_mean']
    x_std = checkpoint['x_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']

    # Instantiate the model and load the state dict
    model = RegressionModel(
        input_size=checkpoint['input_size'], 
        hidden_size=checkpoint['hidden_size']
        )
    model.load_state_dict(checkpoint)

    # Input features is a list of floats. We have to convert it to tensor of the correct shape
    x = torch.tensor(input_features).unsqueeze(0)

    # Now we have to do the same normalization we did when training:
    x = ((x - x_mean)/x_std)

    # We get the output of the model and we print it
    output = model(x)

    # We have to revert the target normalization that we did when training:
    output = output * y_std + y_mean
    print(f"The predicted price is: ${output.item()*1000:.2f}")
