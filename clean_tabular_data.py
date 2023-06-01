import pandas as pd

class DataCleaning:

    def __init__ (self):
       pass

    def clean_product_data(self):
        #Making the csv file readable with a specific line terminator
        df = pd.read_csv('Products.csv', lineterminator='\n')
        df['price'] = df['price'].str.replace(',', '').str.replace('£', '').astype(float)

        #Extracting root category from the categories column. For example Home & Garden/ Shed would just be Home & Garden
        df['category'] = df['category'].str.partition('/')[0] 
        df['category'] = df['category'].astype('category')

        #Creating an encoder which will come into use later on
        categories = df['category'].unique()
        num_of_categories = list(range(len(categories)))
        encoder = dict(zip(categories,num_of_categories))

        #merging products and images csv's to make one large dataframe and adding a numerical category in a new column called Labels
        df2 = pd.read_csv('Images.csv')
        df3 = df.merge(df2,left_on='id',right_on='product_id') 
        labels, _ = pd.factorize(df3['category'])
        df3['labels'] = labels
        df3.to_csv('training_data.csv')
        print(df3)

        
        
    


x = DataCleaning()
x.clean_product_data()
