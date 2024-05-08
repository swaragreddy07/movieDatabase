# imporing all the required libraries
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from clean_data import dataset

# importing the pickle file
pickle_file_path = 'faiss.pkl'

with open(pickle_file_path, 'rb') as f:
    index = pickle.load(f)

model = SentenceTransformer("distilbert-base-nli-mean-tokens")

# performing search for the user input
def search(query):
    queries = []
    queries.append(query)
    vector  = model.encode(queries)
    faiss.normalize_L2(vector)
    dist, ann = index.search(vector, k=7)
    return ann
    
st.header("Welcome to the movies Database")

query = st.text_input("Enter your movie search query:")

# printing the resuts of the search

if st.button("search"):
    if query:
      indices = search(query)
      for i in range(0, len(indices[0])):
          with st.container():
            col1, col2 = st.columns([2, 3])  

            with col1:
              st.subheader(dataset["Series_Title"].iloc[indices[0][i]])
 
            with col2:
              st.write(f"**IMDB Rating:** {dataset['IMDB_Rating'].iloc[indices[0][i]]}")
              st.write(f"**Genre:** {dataset['Genre'].iloc[indices[0][i]]}")
              st.write(f"**Director:** {dataset['Director'].iloc[indices[0][i]]}")
              st.write(f"**Star:** {dataset['Star1'].iloc[indices[0][i]]}")
              st.write(f"**Overview:** {dataset['Overview'].iloc[indices[0][i]]}")
          st.markdown("---")
        
        

          
