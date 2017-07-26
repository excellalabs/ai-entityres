import tensorflow as tf
import numpy as np
import re
from nltk.tokenize import WhitespaceTokenizer
import pickle

import network_modules as nm
from simulated_swarm import simulated_swarm as ss
import padding

class DMN:
    def __init__(self, vec_dim):
        '''vec_dim is the dimension of the word embeddings
        output_dim is the number of possible hyperlinks'''
        self.vec_dim = vec_dim
        self.learning_rate = 1e-3
        self.velocity = 1e-1
        self.repel = 1e-1
        self.decay_rate = 0.05
        self.jump_size = 5.
        
        
    def toTokens(self,text):
        retval = []
        tokens = WhitespaceTokenizer().tokenize(text)
        for t in tokens:
            t = t.lower()
            t = re.sub(r'\W+', '', t)
            if len(t)<1:
                continue
            retval.append(t)
        if len(retval) == 0:
            retval = "NONE"
        else:
            retval = retval[0]
        return retval

    def train(self, data):
        ## Encode the keywords and link text as variables
        print('Beginning to embed words as variables.')
        keywords = []
        linkwords = []
        urls = []
        for i,row in enumerate(data):
            for word in row[0]:
                keywords.append(self.toTokens(word))
            linkwords.append(row[1])
            urls.append(row[2])
            if (i+1) % 1000 == 0:
                print('Tabulated data in', i+1, 'rows, out of', len(data))
        print('Writing variable embedding dictionary.')
        self.embedding_keys = {word:tf.Variable(tf.random_uniform([self.vec_dim], -1.,1.),
                                                                   dtype=tf.float32, name = 'key')
                                    for word in set(keywords)}
        self.embedding_links = {word:tf.Variable(tf.random_uniform([self.vec_dim], -1.,1.),
                                                                   dtype=tf.float32, name = 'link')
                                    for word in set(linkwords)}
        self.output_dim = len(set(urls))
        url_dict = {url:np.zeros(self.output_dim) for url in set(urls)}
        for i,url in enumerate(url_dict.keys()):
            url_dict[url][i] = 1.    
        print("Completed writing variable embedding dictionaries.")
        
        data0 = []
        data1 = []
        data2 = []
        for i,row in enumerate(data):
            word = []
            for elem in row[0]:
                word.append(self.embedding_keys[self.toTokens(elem)])
            data0.append(word)
            data1.append(self.embedding_links[row[1]])
            data2.append(url_dict[row[2]])
            if (i+1) % 1000 == 0:
                print('Converted', i+1, 'row in data to TF variables, out of', len(data))
        self.f = tf.stack(padding.padding_vec(data0,self.vec_dim)) ## fact vectors
        self.q = tf.stack(data1)                                   ## question vectors
        data2 = np.array(data2)
        data2_place = tf.placeholder(shape = data2.shape, dtype = tf.float32)
        print('Completed stacking data.')
        
        self.answer = nm.answer(self.f, self.q, self.vec_dim, self.output_dim, name = 'epi')
        self.answer.get_answer()
        self.a = self.answer.answer
        print('Model built.')
        
        self.loss = - tf.reduce_mean(tf.tensordot(data2_place,tf.log(self.a + 1e-8),axes=[[1],[1]]))
        print('Loss built.')
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)   
        print('Train ops built.')
        
        print('Beginning training.')
        ss(self.loss, self.train, name = 'dmn', velocity = self.velocity,
           decay_rate = self.decay_rate, jump_size = self.jump_size,
           feed_train = {data2_place : data2}, feed_eval = {data2_place : data2}, epochs = 10000)
        
        correct_prediction = tf.equal(tf.argmax(data2_place, 1), tf.argmax(self.a, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        with tf.Session() as sess:
            tf.train.Saver().restore(sess,'./variables/dmn.ckpt')
            self.word_dict = sess.run(self.embedding_keys)
            for phrase,val in self.embedding_links.items():
                self.word_dict[phrase] = sess.run(val)
            self.train_accuracy = accuracy.eval(feed_dict={data2_place : data2})
        
        with open('./data/dict.pkl','wb') as file:
            pickle.dump(self.word_dict,file)
        
        with open('./data/accuracy.txt','w') as file:
            file.write(str(self.train_accuracy))