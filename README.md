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
