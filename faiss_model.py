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