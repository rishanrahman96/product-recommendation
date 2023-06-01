# facebook_product_recommendations

## Part 1: Retrieving the data

The dataset for this task was stored in an EC2 instance in AWS. To access this, I needed to SSH into it. To do this, I needed to download the private key from the S3 bucket. After configuring the connection to the EC2 and SSH'ing into it, there are two csv files and a folder containing product images. The two csv files are products.csv and images.csv. They contain information about the products such as the price, location, description, id etc whilst image.csv contains information about the images, such as their image and product id. This will be used in the data cleaning step.

## Part 2: Cleaning the data

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

## Part 3: Creating the pytorch dataset

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
