import tensorflow as tf
import pickle

def simulated_swarm(tfloss, tftrain, name, velocity = 1e-1, repel = 1e-1, decay_rate = 0.05, jump_size = 5, jump_trigger = 50, jumps_max = 20, epochs = 10000, feed_train = None, feed_eval = None, resume = False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if resume:
            saver.restore(sess, './variables/'+name+'.ckpt')
        
        print('There are', len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), 'variables to be trained.')
        var_val0 = {elem.op.name : sess.run(elem) for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
        var_val1 = var_val0
        best_var = var_val0
        
        print('Running initial loss computation.')
        new_loss = sess.run(tfloss, feed_dict = feed_eval)
        best_loss = new_loss
        old_loss = new_loss
        print("Initial loss is:", new_loss)
        
        jumps = 0
        consec_worse = 0
        best_epoch = -1
        
        loss_set = [new_loss]
        for epoch in range(epochs):
            sess.run(tftrain, feed_dict = feed_train)
            new_loss = sess.run(tfloss, feed_dict = feed_eval)
            
            loss_set.append(new_loss)
            
            if new_loss < best_loss:
                best_loss = new_loss
                best_epoch = epoch
                if (epoch + 1) % 250 == 0:
                    saver.save(sess, './variables/'+name+'.ckpt')
                    best_var = {elem.op.name : sess.run(elem) for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}

                else:
                    best_var = {elem.op.name : sess.run(elem) for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
                var_val1 = {elem.op.name : sess.run(elem) for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
                consec_worse = 0
                
            if new_loss/old_loss - 1. < 0.001:
                consec_worse += 1
                
            print("Training epoch no.:", epoch+1)
            print("Current loss:", new_loss)
            print("Best loss is:", best_loss)
            print("Best epoch is:", best_epoch+1)
            print("The jump is: ", jumps)
            print("\n")
            
            
            if consec_worse >= jump_trigger:
                for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    sess.run(tf.assign(elem, elem
                                       + tf.random_normal(elem.shape,
                                                          mean=0.,
                                                          stddev=jump_size)
                                       + velocity * (elem - var_val0[elem.op.name]))
                                       + repel * (elem - var_val1[elem.op.name]))
                var_val0 = {elem.op.name : sess.run(elem) for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
                jump_size *= decay_rate
                
                jumps += 1
                consec_worse = 0
            
            if jumps > jumps_max:
                break
            
            old_loss = new_loss
        for elem in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            sess.run(tf.assign(elem, best_var[elem.op.name]))
        saver.save(sess, './variables/'+name+'.ckpt')
        
        with open('./data/loss_set.pkl','wb') as file:
            pickle.dump(loss_set,file)
        print("Training Complete.")