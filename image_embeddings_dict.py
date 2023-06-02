from pytorch_dataset_custom import Images
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from neural_network import Neural_networkN
import pickle

#Initialising the images and neural network classes
dataset = Images()
model = Neural_networkN()

#Splitting the data and loading the training data into a dataloader
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

#loading in the weights from the feature extraction model run and setting model to eval mode
weights_path = '/Users/rishanrahman/Desktop/product-recommendation/model_evaluation_full/2023-06-01_16-32-15/weights/epoch_2.pt'
model.load_state_dict(torch.load(weights_path))
model.eval()
#Initialising an empty dictionary to store the image_id's and embeddings
image_embedding = {}

#Creates an iterator which allows to iterate over batches
iterator = iter(train_dataloader)

#Code wrapped in a try block, so when there is no "next" it will save pickle file
try:
    while next(iterator):
        #Extracting features,labels,image_id from each batch
        features,labels,image_id = next(iterator)
        with torch.no_grad():
            #Running the model to get embeddings for each feature
            embedding = model(features)
            #Creating dictionary which is used to store id's and embeddings
            image_embedding.update(zip(image_id,embedding))
except StopIteration:
    #Saving pkl file
    with open('image_embedding.pkl', 'wb') as fp:
        pickle.dump(image_embedding, fp)
