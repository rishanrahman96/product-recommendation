# facebook_product_recommendations

## Part 1: Retrieving The Data

The dataset for this task was stored in an EC2 instance in AWS. To access this, I needed to SSH into it. To do this, I needed to download the private key from the S3 bucket. After configuring the connection to the EC2 and SSH'ing into it, there are two csv files and a folder containing product images. The two csv files are products.csv and images.csv. They contain information about the products such as the price, location, description, id etc whilst image.csv contains information about the images, such as their image and product id. This will be used in the data cleaning step.

## Part 2: Cleaning The Data

The images.csv file has an id column which matches with the product_id column in the products.csv file. The products.csv also has a columns category which is used to apply a category to. For example "Home & Garden" would correspond to category 0 or "Appliances" which would correspond to category 1. To do this, we needed to extract the root category from the categories column, which can be done with a simple string method as shown below:

```python
 df['category'] = df['category'].str.partition('/')[0] 
 df['category'] = df['category'].astype('category')
```
One of the main purposes of this is to create an image classification model based on these images, for that we needed training data which provided labels, as well as a way to access the images in the file. To do this we needed the image_id's. Firstly, we joined the two csv files we have and then created a new column which provided a numerical encoding for the categories. This was done using the following lines of code:

```python
 df2 = pd.read_csv('Images.csv')
 df3 = df.merge(df2,left_on='id',right_on='product_id') 
 labels, _ = pd.factorize(df3['category'])
 df3['labels'] = labels
 df3.to_csv('training_data.csv')
```

We also created an encoder for the categories which would be used later in the project to decode the predictions:

```python
 df2 = pd.read_csv('Images.csv')
 df3 = df.merge(df2,left_on='id',right_on='product_id') 
 labels, _ = pd.factorize(df3['category'])
 df3['labels'] = labels
 df3.to_csv('training_data.csv')
```
The data was also cleaned of rogue ',' in columns and £ signs.

The Images folder from the EC2 instance also had to be cleaned in prep for deep learning. We needed to make sure that all the images were consistent, i.e the same size, the same number of channels. To do this, clean_images.py was created to process the images.

## Part 3: Creating The PyTorch Dataset

Next up, we created a custom pytorch dataset. This readied the data to be used in machine learning models. For example, you can't just feed a machine learning model an image. It needs to be changed in a way that is understood by computers. The way to do this is to convert everything into torch tensors and pytorch is a framework which makes this super easy.

```python

class Images(Dataset):

    def __init__(self):
        super().__init__()
        #Load in the dataset and assign to a dataframe
        self.df = pd.read_csv('training_data.csv', lineterminator='\n')
        #Add transforms to the data which will allow it to be used in future models.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) 
        ])

    def __getitem__(self, index):
            #assigning labels as features for the final dataset which will be used to train the model
            labels = self.df.loc[index,'labels']
            image_id = self.df.loc[index,'id_y']
            file_path = f'clean_images/{image_id}.jpg'
            with Image.open(file_path) as img:
                img.load()
            features = self.transform(img)
            return features, labels
    
    def __len__(self):    
        return len(self.df)
```
This Images class is then later used to define the dataset. Which will be used in the training process of the model.

## Part 4: Creating The Model & Transfer Learning

The whole point of this project is to return recommendations based on a certain product. To do that, we're first going to create an image classification model. As we're dealing with images, it is useful to use a deep learning model here. To do so, I used a convolutional neural network, however instead of custom making my own (a huge and time consuming task), I made the most of existing models. 

In this project, I used the resnet50 which performs a 1000 way classification on the imagenet dataset. So, you might think great, let's move on, but we can't quite do that yet. This is because resnet50 is not finetuned to our dataset. To do this, we used transfer learning to freeze all but the last two layers of the model. You can see this in the code below:

```python

class Neural_networkN(torch.nn.Module):
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
        return x
```

## Part 5: Creating the training loop

Now that we have created our model and our dataset, we can use a training loop to achieve the weights of our model which shall be used in the feature extraction stage. To do this, we loop through our train dataloader, for each batch we make a prediction which is then used to calculate a loss. This is then backpropogated through the neural network. A training and validation loss is calculated for each epoch along with a validation accuracy. This, combined with our tensorboard visualisations are used to decide which epoch to use for the model weights. The weights are also stored in a folder called model evaluation. Initially, we reduced the size of our image to 64 x 64 in the custom pytorch dataset. This is because CUDA is not available on m1 macbook pros and it's own metal gpu does not increase the speed of computation enough. By lowering the image size, the task is less computationally intensive. We can then warm start the model again using the desired size and preloading the weights from before.

```python

dataset = Images()
model = Neural_networkN()

# Split the dataset into training and test sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Creating dataloaders for all sets of data
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

#Creating the training loop
def train(model, train_dataloader, val_dataloader, epochs = 10):
    #initialising empty lists to keep track of the various losses
    train_losses = []
    valid_losses = []
    valid_accuracy = []
    #defining an optimiser for the training loop
    optimiser = torch.optim.Adam(model.parameters(),lr =0.001)
    #Using summary writer which creates logs in the working directory for tensorboard visualisations
    writer_train = SummaryWriter(log_dir='logs/train')  
    writer_val = SummaryWriter(log_dir='logs/val')  

    batch_idx =0

    # Create a folder for the model evaluation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_eval_folder = f'model_evaluation_loaded_weight/{timestamp}'
    os.makedirs(model_eval_folder, exist_ok=True)

    # Create a folder for the model weights
    weights_folder = os.path.join(model_eval_folder, 'weights')
    os.makedirs(weights_folder, exist_ok=True)

    #Looping through each epoch
    for epoch in range(epochs):

        train_loss = 0.0
        valid_loss = 0.0
        #Setting the model to train to allow backpropogation
        model.train()
        #Adding a progress bar to monitor time and speed per iteration
        pbar = tqdm(train_dataloader)
        #For each batch, features and labels are unpacked and a prediciton is made with features which is used to calculate a loss that is backpropogated
        for batch in pbar:
            features,labels= batch
            prediction = model(features)
            loss = F.cross_entropy(prediction,labels)
            loss.backward() #does not overwrite, adds to what is already there, need to manually reset gradient to 0 everytime we go through the loop
            pbar.set_description(f'{epoch},{loss.item()}')
            optimiser.step()
            optimiser.zero_grad()
            writer_train.add_scalar('loss',loss.item(),batch_idx)
            train_loss += loss.item() * features.size(0)
            batch_idx += 1
        
        total_t = 0
        correct_t = 0
        #Model changed to eval mode to prevent back propogation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_outputs = model(val_features)
                val_loss = F.cross_entropy(val_outputs, val_labels).item()
                writer_val.add_scalar('loss', val_loss, epoch)
                valid_loss += val_loss * val_features.size(0)

                _,pred_t = torch.max(val_outputs,dim=1)
                correct_t += torch.sum(pred_t == val_labels).item()
                total_t += val_labels.size(0)
            validation_accuracy = 100*correct_t/total_t
            valid_accuracy.append(validation_accuracy)

        train_loss = train_loss/len(train_dataloader.sampler)
        valid_loss = valid_loss/len(val_dataloader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        print(f'Validation accuracy: {validation_accuracy}')


        # Save model weights
        save_path = os.path.join(weights_folder, f'epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), save_path)

        # Save model metrics
        metrics = {
            'epoch': epoch + 1,
            'validation_loss': val_loss,
        }
        metrics_path = os.path.join(model_eval_folder, f'epoch_{epoch+1}_metrics.pt')
        torch.save(metrics, metrics_path)

    writer_train.close() 
    writer_val.close()

#Weights of previous run loaded in.
weights_path = '/Users/rishanrahman/Desktop/aicore-fb-project/model_evaluation1/2023-06-01_13-30-50/weights1/epoch_11.pt'
model.load_state_dict(torch.load(weights_path))
train(model, train_dataloader, val_dataloader, epochs=10)

```

## Part 6: Extracting High Level Features

Our previous task created an image classification model, but for our end goal of comparing similar products, this is not actually what we need. In reality, we need to extract high level features from each image, i.e we want a vector to represent each image. The reason we do the image classification to start with is to achieve better vector embeddings for each image (Remember the resnet50 is not finetuned to our task). If we remove the last two fully connected layers of the neural network, we're left with just this.


```python

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


```
This provides a vector with 1000 rows, containing high level features of each image which can be used for FAISS search.

## Part 6: Retrieving Image Embeddings

We now have a feature extraction model which provides us with image embeddings, however we need to do this for all the images in our training dataloader. To do so, we initialised the neural network class again and loaded in the weights from our feature extraction model. We then iterated through our train dataloader to get a dictionary containing the image_id and embedding for each image. The code for this can be found below:


```python

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
```

## Part 7: FAISS Search

The previous stage ended up with us saving a pickle file containing the image_id's and embeddings. The first stage was to load that in as a variable. We then converted the image_id's and embeddings into separate arrays. There was a problem with the shape of the arrays so code was added to ensure they were all homogenous.

```python

import pickle
import faiss
import numpy as np

# Specify the path to your pickle file
pickle_file_path = '/Users/rishanrahman/Desktop/product-recommendation/image_embedding.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    embeddings_dict = pickle.load(file)

# Convert the embeddings dictionary into separate arrays
ids = list(embeddings_dict.keys())
embeddings = list(embeddings_dict.values())

# Determine the maximum shape among all embeddings
max_shape = max([embedding.shape for embedding in embeddings])

# Create a new array to hold the padded embeddings
embeddings_array = np.zeros((len(embeddings),) + max_shape).astype('float32')

# Pad or reshape the embeddings to have a consistent shape
for i, embedding in enumerate(embeddings):
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

# Example query embedding
query_embedding = np.random.randn(1, embeddings_array.shape[1]).astype('float32')


# Perform a vector search
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Retrieve the image IDs and corresponding embeddings of the nearest neighbors
nearest_neighbors_ids = [ids[i] for i in indices[0]]
nearest_neighbors_embeddings = [embeddings[i] for i in indices[0]]

print(nearest_neighbors_ids)

```

## Part 8: Deployment

The fast stage deployment was to create a FAST API. The big advantages of using FAST API was it's speed compared to other frameworks such as Django and Flask, fast development speeds as well as asynchronous code. It also automatically produces documentation which can be used to easily test the API. Below is my API code.

```python

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

class TextItem(BaseModel):
    text: str

try:
    model = Neural_network()
    weights_path = 'epoch_10.pt'
    model.load_state_dict(torch.load(weights_path))
    model = torch.nn.Sequential(*list(model.children())[:-2])

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

```

The api involved two post methods, one for extracting the features of the images and the other for returning the 5 nearest image_id's to the image uploaded. This API was then combined with a docker file and a requirements.txt file to build a docker image that was pulled from an EC2 instance.
