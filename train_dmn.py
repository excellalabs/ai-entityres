import pickle

from dmn import DMN

with open('./data/data.pkl','rb') as file:
    data = pickle.load(file)
    
dmn = DMN(3)
dmn.train(data[:1000])

with open('./data/word_dict.pkl','wb') as file:
    pickle.dump(dmn.word_dict, file)