import tensorflow as tf

##############################################################################
'''GRUt Class'''
##############################################################################
class GRUt:
    '''This is the GRUt class. Contains the hyperparameters and state updating function.

    ** hid_dim specifies the dimension of the output of the hidden layer
       e.g. Episodic memory vector, want the number of possible answers.

    ** vec_dim specifies the dimension of the input vector
       e.g. Context vectors in the memory update mechanism are the dimension of
       the sentence embedding

    ** Usage: initialize the gru object - this will be a layer in the network
       Update the hidden layer output state (successively), self.h, by feeding
       self.update_state a vector (rank 1 tensor) of dimension vec_dim

     ** Number of hyperparameters for m*n*d hid_dim, vec_dim, neuron_no:
         (d*m+1)(m+n+1) + d'''
    def __init__(self, hid_dim = 100, vec_dim = 100, neuron_no = 3, name = None):
        self.hid_dim = hid_dim                  #Dimension of the hidden layer
        self.vec_dim = vec_dim                  #Dimension of the input layer
        self.neuron_no = neuron_no              #Number of neurons in the GRUt
        self.h = tf.zeros(shape = [1, self.hid_dim]) #This is the internal state vector
        # Declare the variables
        self.W_ux = tf.Variable(tf.random_normal([self.vec_dim,1], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".GRUt.W_ux")
        self.W_hx = tf.Variable(tf.random_normal([self.neuron_no, self.vec_dim, self.hid_dim]),
                                dtype= tf.float32, name = name + ".GRUt.W_hx")
        self.W_uh = tf.Variable(tf.random_normal([self.hid_dim,1], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".GRUt.W_uh")
        self.W_hh = tf.Variable(tf.random_normal([self.neuron_no, self.hid_dim, self.hid_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".GRUt.W_hh")
        self.W_h  = tf.Variable(tf.random_normal([self.neuron_no], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".GRUt.W_h")
        self.b_u  = tf.Variable(tf.random_normal([1], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".GRUt.b_u")
        self.b_h  = tf.Variable(tf.random_normal([self.neuron_no, self.hid_dim], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".GRUt.b_h")


    ## Update the hidden state
    def update_state(self, x):
        ## Update gate
        self.u = tf.tanh(tf.matmul(x, self.W_ux) + tf.matmul(self.h, self.W_uh) + self.b_u) + 1.
        ## Hidden state
        self.hp = self.h
        self.h = tf.tensordot(self.W_h, tf.tanh(tf.tensordot(x, self.W_hx, axes=[[1],[1]])
                 +  tf.tensordot(self.hp, self.W_hh, axes= [[1],[1]]) + self.b_h), axes = [[0],[1]])
        self.h = self.u * self.h + (1. - self.u) * self.hp
        
##############################################################################
'''Attention GRU Class'''
##############################################################################
class attnGRU:
    '''This is the Attention GRU class. Contains the hyperparameters and state
       updating function.

    ** hid_dim specifies the dimension of the output of the hidden layer
       e.g. Number of context sentence vectors in the .

    ** vec_dim specifies the dimension of the input vector
       e.g. Context vectors in the memory update mechanism are the dimension of
       the sentence embedding

    ** trainQ specifies if the GRU is being trained. Randomly initializes the
       hyperparameters

    ** Usage: Initialize the attention GRU by specifying the input and output shapes
       Attention GRUs generate a final state by running on a sequence of vectors (rank 1 tensors).
       There is a gate associated with each element of the sequence.
       To acquire the attention gates, feed a "rank 2 tensor" of shape
       [length of sequence, vec_dim].
       To update the object's state on the next element of the sequence, run self.update_state().
       NOTE: The scare quotes around "rank 2 tensor" implies this is not really a rank 2 tensor;
       the first index of the object is interpreted as a sequence/tuple index, while the second
       index is actually a vector index - this does not transform as a tensor.'''
    def __init__(self, x, hid_dim, vec_dim, name = None):
        '''x is the set of concatenated input, {z_i}. Needs to be a list of vectors'''
        self.x = x
        self.x_t = tf.transpose(x, perm = [1,0,2])
        self.shape = tf.shape(x)
        self.hid_dim = hid_dim
        self.vec_dim = vec_dim
        self.h = tf.zeros(shape = [self.shape[0], self.hid_dim])
        # Declare the variables
        self.W_Zx = tf.Variable(tf.random_normal([hid_dim, vec_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_Zx")
        self.W_Zz = tf.Variable(tf.random_normal([hid_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_Zz")
        self.W_rx = tf.Variable(tf.random_normal([hid_dim, vec_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_rx")
        self.W_rh = tf.Variable(tf.random_normal([hid_dim, hid_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_rh")
        self.W_hx = tf.Variable(tf.random_normal([hid_dim, vec_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_hx")
        self.W_hh = tf.Variable(tf.random_normal([hid_dim, hid_dim], mean= 1., stddev= 3.), dtype= tf.float32, name = name + ".aGRU.W_hh")
        self.W_h  = tf.Variable(tf.random_normal([1], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".aGRU.W_h")
        self.b_x  = tf.Variable(tf.random_normal([hid_dim], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".aGRU.b_x")
        self.b_z  = tf.Variable(tf.random_normal([1], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".aGRU.b_z")
        self.b_r  = tf.Variable(tf.random_normal([hid_dim], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".aGRU.b_r")
        self.b_h  = tf.Variable(tf.random_normal([hid_dim], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".aGRU.b_h")


    def get_gates(self):
        self.mask = tf.cast(tf.not_equal(tf.reduce_sum(tf.abs(self.x), 2), 0), tf.float32)
        self.Z = (tf.einsum('i,jki->jk', self.W_Zz, tf.tanh(tf.einsum('ijk,lk->ijl',self.x, self.W_Zx)
                            + self.b_x)) + self.b_z)
        self.Z = tf.where(tf.equal(self.mask, 0), tf.ones_like(self.mask) * (-float("inf")), self.Z)
        self.g = tf.nn.softmax(self.Z)
        self.g = tf.transpose(tf.reshape(tf.tile(self.g, [self.hid_dim,1]), shape = [self.hid_dim, self.shape[0], self.shape[1]]),
                              perm = [1,2,0])
        self.g_t = tf.transpose(self.g, perm = [1,0,2])



    ## Function to update the hidden state (Single RNN layer)
    def update_state(self, x, g, h):
        r = (tf.tanh(tf.einsum('ij,kj->ik',x, self.W_rx)
                  + tf.einsum('ij,kj->ik',h, self.W_rh)
                  + self.b_r) + 1.)
        hp = h
        h = self.W_h * (tf.tanh(tf.einsum('ij,kj->ik',x, self.W_hx)
                  + r * tf.einsum('ij,kj->ik',hp, self.W_hh) + self.b_h))
        return (g * h + (1. - g) * hp)

    # Function to obtain states from all RNN layers
    def get_states(self):
        self.h_set = tf.transpose(tf.scan(lambda h, x: self.update_state(x[0], x[1], h), (self.x_t, self.g_t), self.h),
                              perm = [1,0,2])
        self.h = self.h_set[:,self.shape[1]-1]
        
##############################################################################
'''Episodic Memory Class'''
##############################################################################
class episodic_memory:
    '''This is the episodic memory class. Contains the hyperparameters and state
       updating function. Contains the memory vector and context vector as attributes.

    ** set_length is the length of the sequence over which attention is to be paid

    ** csent is the set of contextual facts relevant to the module, supplied
       as a "rank 2 tensor" of shape [set_length, semantic_space_dimensions]

    ** qsent is the question vector, a rank 1 tensor of shape [semantic_space_dimensions]

    ** episodes specifies how many episodes to run the memory over before
       accepting the memory and context vectors as

    ** self.h is the context vector; self.memory is the memory vector obtained
       after running episodes

    ** trainQ specifies if the GRU is being trained. Randomly initializes the
       hyperparameters

    ** Usage: Initialize the object by supplying the set length, set of contextual facts,
       question vector, and number of episodes to run.
       Obtain the final memory and context vectors by running self.run_episodes()'''
    def __init__(self, csent, qsent, vec_dim, name = None, episodes = 3):
        ## Set up the GRU and attention GRU
        self.q = qsent  #Question vector
        self.f = csent  #Fact vectors
        self.f_t = tf.transpose(csent, perm = [1,0,2])
        self.vec_dim = vec_dim
        self.x = tf.transpose(tf.map_fn(lambda y: tf.concat([y * self.q, y * self.q, y - self.q, y - self.q], axis = 1),
                                        self.f_t), perm = [1, 0, 2])

        self.gru  = GRUt(hid_dim = vec_dim, vec_dim = vec_dim, name = name)           #Make GRU to update the memory vector
        self.agru = attnGRU(x = self.x, hid_dim =  vec_dim, vec_dim = 4*vec_dim, name = name)    #Make the attention GRU for the context vector
        self.episodes = episodes #Number of episodes to run the module over
        self.gru.h = self.q      #Initialize the memory vector to the question

    def update_state(self):
        ## Compute the context vector first
        self.agru.get_gates()         # Get the gates
        self.agru.get_states()        # Run the attention gru over all inputs
        self.h = self.agru.h          # Get the context vector
        self.agru.h = tf.zeros(shape = [tf.shape(self.q)[0], self.vec_dim]) # Reset the aGRU in prep for next episode
        self.agru.gate_ind = 0

        ## Update the memory vector
        self.gru.update_state(self.h)
        ## Update the features
        self.x = tf.transpose(tf.map_fn(lambda y: tf.concat([y * self.gru.h, y * self.q, y - self.gru.h, y - self.q], axis = 1),
                                        self.f_t), perm = [1, 0, 2])

    def run_episodes(self):
        for ind in range(self.episodes):
            self.update_state()
        self.memory = self.gru.h
        
##############################################################################
'''Weights Class'''
##############################################################################
class weights:
    '''Answer module class. This takes a semantic vector, x, and an ordered set of
       candidate answers (as a "rank 2 tensor" of shape [length of candidate set, semantic dimensions]
       and returns the ordered probabilities of candidate answers.'''

    def __init__(self, x, vec_dim, output_dim, name = None):
        self.x = x
        self.vec_dim = vec_dim
        self.hid_dim = output_dim
        # Declare variables
        self.W = tf.Variable(tf.random_normal(shape = [self.hid_dim, self.vec_dim], mean= 1., stddev= 3.), dtype = tf.float32, name = name + ".weights.W")

        self.a = tf.nn.softmax(tf.einsum('ij,kj->ik',self.x,self.W))
        
class answer:
    def __init__(self, csent, qsent, vec_dim, output_dim, episodes = 3, name = None):
        self.f = csent
        self.f_t = tf.transpose(csent, perm = [2,0,1])
        self.q = qsent
        self.vec_dim = vec_dim
        self.output_dim = output_dim
        self.name = name
        self.epi = episodic_memory(csent = csent, qsent = qsent, vec_dim = vec_dim, episodes = episodes, name = name)

    def get_answer(self):
        self.epi.run_episodes()
        self.weights = weights(x = self.epi.memory, output_dim = self.output_dim,
                               vec_dim = self.vec_dim, name = self.name)
        self.answer = self.weights.a