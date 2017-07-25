# AI Application for Entity Resolution

**Question**
Build an AI (neural network) application using Google's dataset for document entity resolution. Provide the code via GIT for review, login information to view a dashboard of statistics describing the application’s performance (i.e. precision and recall, combined MAP rate), and some method of visualization that provides meaningful comprehension and interpretability to conceptual clusters and other data relevant towards understanding features of entities and data quality. Please provide the government with the credentials to access the code for review and testing purposes.

**Response**
The AI creation challenge calls for the entity resolution of words.  The entity resolution of the meaning of words in a given context is a dynamic classification problem using natural language processing (NLP) whereby a word is mapped to a unique definition for that word when provided the contextual words that surround it.  The Google documents referenced in the challenge label the definition of a word of interest by providing a link to the Wikipedia page associated with the word in that particular context.  To this end, the inputs to the neural network must be the word of interest and surrounding words; and output of the neural network should be the URL to the Wikipedia page labeling that definition.   
 
NLP with neural networks is most performant when mathematical representations of words, known as word embeddings, are fed into the network instead of the ASCII language itself.  Word embeddings are vectors, or ordered sets of numbers, that are constructed by either statistical means or by another neural network.  Since the corpus is very large, and Excella's Lie Group Embeddings (LiGrE) is more computationally expensive than GloVe, we employ GloVe embeddings, as discussed in section 6. Since the GloVe scheme encodes the statistical likelihood two words will be used in a similar context, the neural network is primed to identify the meaning of a word given context. 
 
The technical challenge in this classification problem arises from the fact that the number of labels used for classification varies by the word to be resolved.  Such problems are aptly addressed using a neural network architecture known as a "dynamic memory network," or DMN.  DMNs excel at picking a specific answer given an arbitrary set of possible answers by identifying the most relevant words in a context and mapping them to the correct answer.  In this case, the question is "which of the available definitions does the word take on in the provided context?" 
 
The provided neural network and code is trained on the Google document that contains a set of words, the sentences in which the words are used, and the links to the Wikipedia pages associated with the words in their context.  The network consists of an input layer that converts the words to GloVE vectors and makes a call to a table housing the number of definitions associated with the word of interest; the DMN that determines which definition is appropriately associated with the word given the context; and an output layer that returns the URL.  

After training, the resulting accuracy on a validation set was:

**Visualization of Results**

https://plot.ly/dashboard/psmith1223:7/view
