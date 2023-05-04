import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken


#df = pd.read_csv('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/wholeQuranEmbedded_ordered_wChapterNames.csv')

# Convert the 'embedding' column to a NumPy array
#embedding_array = df['embedding'].apply(eval).apply(np.array).values

# Save the NumPy array to a file
#np.save('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/embedding_array.npy', embedding_array)
# embedding_array = np.load('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/embedding_array.npy',allow_pickle=True)

# df['embedding_array'] = embedding_array.tolist()
# Extract the numpy array from the DataFrame column
# data = df['embedding'].values

# # Save the numpy array to disk as an .npy file
# np.save('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/embedding.npy', data)



# # Load the numpy array from the saved file
# data = np.load('/Users/adilqaisar/Documents/islam finder app/embeddedFiles/embedding.npy',allow_pickle=True)
# new_df = df.assign(new_column=data)
# new_df = pd.DataFrame({'new_column': data})

# # Concatenate existing dataframe and new dataframe
# result_df = pd.concat([df, new_df], axis=1)

# # The 'new_column' column is now a numpy array
# print(type(result_df['new_column'])) # <class 'numpy.ndarray'>

api_key = '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'

openai.api_key= api_key


# search_term_vector = get_embedding("relationships", engine="text-embedding-ada-002")

# df["similarities"] = df['embedding_array'].apply(lambda x: cosine_similarity(x, search_term_vector))
# sorted_by_similarity = df.sort_values("similarities", ascending=False).head(10)

# print(sorted_by_similarity.head(10))
dfs = []

filename = "/Users/adilqaisar/Documents/Zacks/AAPLEER.csv"
df = pd.read_csv(filename)
    

df['embedding'] = df['sentences'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
dfs.append(df)
print(type(df['embedding'][0]))

 

# Concatenate the DataFrames into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv("AAPLEER_embedded.csv", index=False)