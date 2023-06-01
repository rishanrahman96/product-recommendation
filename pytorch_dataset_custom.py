from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image


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
