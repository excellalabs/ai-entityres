To unpickle: 

import pickle

with open('dict.pkl', 'rb') as file:
    vectDict = pickle.load(file)
