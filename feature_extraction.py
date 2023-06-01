from neural_network import Neural_networkN
import torch
import os

model = Neural_networkN()
weights_path = '/Users/rishanrahman/Desktop/product-recommendation/model_evaluation_full/2023-06-01_16-32-15/weights/epoch_2.pt'
model.load_state_dict(torch.load(weights_path))
model = torch.nn.Sequential(*list(model.children())[:-2])

save_path = 'final_model/image_model.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)