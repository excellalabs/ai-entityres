# AI Application for Entity Resolution

**Question**
Build an AI (neural network) application using Google's dataset for document entity resolution. Provide the code via GIT for review, login information to view a dashboard of statistics describing the application’s performance (i.e. precision and recall, combined MAP rate), and some method of visualization that provides meaningful comprehension and interpretability to conceptual clusters and other data relevant towards understanding features of entities and data quality. Please provide the government with the credentials to access the code for review and testing purposes.

**Response**
The AI creation challenge calls for the entity resolution of words; that is, the meaning of an isolated word may be ambiguous, due either to the degenerate definitions of the word, or by the existence of synonyms.  The entity resolution of the meaning of words in a given context is a dynamic classification problem using natural language processing (NLP), whereby a word is mapped to a unique definition for that word when provided the contextual words that surround it. The Google documents referenced in the challenge label the definition of a word of interest by providing a link to the Wikipedia page associated with the word in that particular context. To this end, the inputs to the neural network must be the word of interest and surrounding words; and output of the neural network should be the URL to the Wikipedia page labeling that definition.

NLP with neural networks is most performant when mathematical representations of words, known as word embeddings, are fed into the network instead of the ASCII language itself. Word embeddings are vectors, or ordered sets of numbers, that are constructed by either statistical means or by another neural network. Excella's Lie Group Embedding (LiGrE) scheme is built around preserving the contextual meaning of words while enforcing the statistical demands made in the industry-famed GloVe construction.  However, to remain maximally agnostic about an embedding scheme so as to not bias data with human-built models, we sought to create a network to simultaneously classify a word by its definition while constructing the embeddings.

The technical challenge in this classification problem arises from the fact that the number of labels used for classification varies by the word to be resolved. Such problems are aptly addressed using a neural network architecture known as a "dynamic memory network," or DMN. DMNs excel at picking a specific answer given an arbitrary set of possible answers by identifying the most relevant words in a context and mapping them to the correct answer. In this case, the question is "which of the available definitions does the word take on in the provided context?"

To implement the DMN, we utilize TensorFlow.  TensorFlow is a popular, open-source tool for deep learning provided by Google.  It allows one to construct a "computational graph" of tensors and operations to be performed on tensors that is then compiled to C and run.  The biggest values added by TensorFlow is its versitility and algorithms: it can be compiled and run on any platform (CPU, GPU, mobile), and leverages extremely powerful optimization algorithms to train model parameters.  Keras and TFLearn sit on top on TensorFlow to enable quick implementation of the tool, implementing GRUt and our custom episodic memory demanded we work with the core language.

The provided neural network and code is trained on a subset of the provided Google documents. URLs were taken to be the definition of the linked text, and tokens within 200kb distance of the link were taken to be context.  We kept only those data points for which the URL appeared at least 200 times in the corpus. Due to the combined demand of computing power and time to train the task of identifying the correct URL given context and link text on an entire corpus, we considered only 1000 mentions. The set of 2015 combined tokens and linked text were embedded as 3d vectors, and the network trained to select the correct link from a set of 270 unique URLs for 2 hrs, using mean cross entropy as the loss. To measure the performance of our network in surrogate for precision and recall, we monitor and measured the convergence of our cross entropy loss function.   The initial mean cross entropy was ~18, and it reached 8.3 over 5153 epochs.  This represents a ~22000 factor improvement on the predicted probability distribution over the course of training. It is still a fair error, but, given the considerable improvement in the loss over the initial training period, with more training time, this would be substantially improved if allowed to train for longer.

**Visualization of Results**

Visualization of the results was done utilizing Python plotly. The code to generate these visualizations can be found here:

https://plot.ly/dashboard/psmith1223:8/view

**Execution**

To run the model from start to finish, first, run prep_data.py.  Then, simply run train_dmn.

If training is being resumed instead, add the argument "resume = True" to the ss function at like 99 in dmn.py.

The output from the training is stored in the data folder.  Model variables relevant to TensorFlow are stored in the variables folder.
