import numpy as np
import pandas as pd




df = pd.read_csv('/Users/adilqaisar/Documents/Zacks/AAPLEER_embedded.csv')

# Convert the 'embedding' column to a NumPy array
embedding_array = df['embedding'].apply(eval).apply(np.array).values

# Save the NumPy array to a file
np.save('AAPLEER_embedded_array.npy', embedding_array)
#embedding_array = np.load('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/embedding_array.npy',allow_pickle=True)

# Load the original npy file
data = np.load('AAPLEER_embedded_array.npy',allow_pickle=True)

# Save the compressed npz file
np.savez_compressed('compressed_AAPLEER_embedded_array.npz', data=data)