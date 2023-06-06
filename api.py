import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchvision import transforms, models
import  numpy as np
import faiss

##############################################################
# TODO                                                       #
# Import your image processing script here                 #
##############################################################

class Neural_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        for name, module in self.resnet50.named_modules():
            if name == 'layer4' or name == 'fc':
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False


        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(1000, 13)
        self.main = nn.Sequential(self.resnet50, self.linear)
            
    def forward(self, inp):
        x = self.main(inp)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    

    
# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:

    model = Neural_network()
    weights_path = 'epoch_10.pt'
    model.load_state_dict(torch.load(weights_path))
    model = torch.nn.Sequential(*list(model.children())[:-2])

    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:                 
    # Specify the path to your pickle file
    pickle_file_path = 'image_embedding.pkl'

    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        embeddings_dict = pickle.load(file)                                       

except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) 
        ])
    
    features = transform(pil_image).unsqueeze(0)
    embeddings = model(features)
    


    return JSONResponse(content={
    "features": embeddings.detach().numpy().tolist()
    
        })


  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    features = transform(pil_image).unsqueeze(0)
    embeddings = model(features)

    # Specify the path to your pickle file
    pickle_file_path = '/Users/rishanrahman/Desktop/aicore-fb-project/image_embedding.pkl'

    # Load the pickle file
    with open(pickle_file_path, 'rb') as file:
        embeddings_dict = pickle.load(file)

    # Convert the embeddings dictionary into separate arrays
    ids = list(embeddings_dict.keys())
    embeddings_list = list(embeddings_dict.values())

    # Convert the query embedding to a numpy array
    query_embedding = embeddings.detach().numpy()

    # Determine the maximum shape among all embeddings
    max_shape = max(embedding.shape for embedding in embeddings_list)

    # Create a new array to hold the padded embeddings
    embeddings_array = np.zeros((len(embeddings_list),) + max_shape).astype('float32')

    # Pad or reshape the embeddings to have a consistent shape
    for i, embedding in enumerate(embeddings_list):
        if embedding.shape != max_shape:
            # Pad or reshape the embedding to match the maximum shape
            if embedding.ndim == 1:
                embeddings_array[i, :embedding.shape[0]] = embedding
            else:
                embeddings_array[i] = np.reshape(embedding, max_shape)
        else:
            embeddings_array[i] = embedding

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])  # Assuming L2 distance metric

    # Add the embeddings to the index
    index.add(embeddings_array)

    # Perform a vector search
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    # Retrieve the image IDs and corresponding embeddings of the nearest neighbors
    nearest_neighbors_ids = [ids[i] for i in indices[0]]
    nearest_neighbors_embeddings = [embeddings_list[i] for i in indices[0]]

    return JSONResponse(content={
        "similar_index": nearest_neighbors_ids
    })



if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)


    
