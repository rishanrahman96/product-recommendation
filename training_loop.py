from pytorch_dataset_custom import Images
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from neural_network import Neural_networkN
import torch
import os
from tqdm import tqdm
import datetime
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

#importing images and neural network class from neural_network.py and pytorch_dataset_custom.py respectively. You can find them in this project!
dataset = Images()
model = Neural_networkN()

print(dataset[10])
# Split the dataset into training and test sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Creating dataloaders for all sets of data
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

# train_iterator = iter(train_dataloader)
# for _ in range(4):  # Skip the first four elements
#     next(train_iterator)

# fifth_element = next(train_iterator)
# print(fifth_element)


def train(model, train_dataloader, val_dataloader, epochs = 10):
    #empty lists initialised to store the losses
    train_losses = []
    valid_losses = []
    valid_accuracy = []
    optimiser = torch.optim.Adam(model.parameters(),lr =0.001)

    #Summary writers are created to create the training and validation logs for tensorboard visualisation
    writer_train = SummaryWriter(log_dir='logs/train')  
    writer_val = SummaryWriter(log_dir='logs/val') 
    #Keeps track of the current batch during training
    batch_idx =0

    # Create a folder for the model evaluation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_eval_folder = f'model_evaluation_loaded_weight/{timestamp}'
    os.makedirs(model_eval_folder, exist_ok=True)

    # Create a folder within model evaluation for the model weights of each epoch
    weights_folder = os.path.join(model_eval_folder, 'weights')
    os.makedirs(weights_folder, exist_ok=True)

  
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        #Setting the model to training mode
        model.train()
        #Creating a progress bar for when script runs, helps to know how much longer left.
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            #For each batch, a prediction is made and with that a loss calculated. this is propogated backwards and weights are zeroed
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
        #model is changed to eval mode to speed up and so backpropogation does not occur
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

        # print-training/validation-statistics for each epoch
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

weights_path = '/Users/rishanrahman/Desktop/aicore-fb-project/model_evaluation1/2023-06-01_13-30-50/weights1/epoch_11.pt'
model.load_state_dict(torch.load(weights_path))
train(model, train_dataloader, val_dataloader, epochs=10)
