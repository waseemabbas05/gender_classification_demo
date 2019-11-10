import tensorflow as tf
import numpy as np

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        '''x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)'''
        return tf.concat([x1, x2], 3)
    
def gender_classifier(x,is_training):
    p_size = 2; strides = 2
    with tf.name_scope("layer1"):
        cnn_out1 = tf.layers.conv2d(inputs=x, filters=4, \
                                     kernel_size=[3,3], padding='same', activation=None)
        cnn_out1 = tf.layers.conv2d(inputs=cnn_out1, filters=4, \
                                     kernel_size=[3,3], padding='same', activation=None)
        concat_x1 = crop_and_concat(cnn_out1,x)
        pool_out1 = tf.layers.average_pooling2d(inputs=concat_x1, pool_size=[p_size,p_size], strides=strides)
        en_bnorm1 = batch_norm(pool_out1, 7, is_training)
        en_relu1 = tf.nn.leaky_relu(en_bnorm1,name=None,alpha = 0.15)
        
        
    with tf.name_scope("layer2"):
        cnn_out2 = tf.layers.conv2d(inputs=en_relu1, filters=8, \
                                     kernel_size=[3,3], padding='same', activation=None)
        cnn_out2 = tf.layers.conv2d(inputs=cnn_out2, filters=8, \
                                     kernel_size=[3,3], padding='same', activation=None)
        concat_x2 = crop_and_concat(cnn_out2,en_relu1)
        pool_out2 = tf.layers.average_pooling2d(inputs=concat_x2, pool_size=[p_size,p_size], strides=strides)
        en_bnorm2 = batch_norm(pool_out2, 15, is_training)
        en_relu2 = tf.nn.leaky_relu(en_bnorm2,name=None,alpha = 0.15)
        
    with tf.name_scope("layer3"):
        cnn_out3 = tf.layers.conv2d(inputs=pool_out2, filters=16, \
                                     kernel_size=[3,3], padding='same', activation=None)
        cnn_out3 = tf.layers.conv2d(inputs=cnn_out3, filters=16, \
                                     kernel_size=[3,3], padding='same', activation=None)
        concat_x3 = crop_and_concat(cnn_out3,en_relu2)
        pool_out3 = tf.layers.average_pooling2d(inputs=concat_x3, pool_size=[p_size,p_size], strides=strides)
        en_bnorm3 = batch_norm(pool_out3, 31, is_training)
        en_relu3 = tf.nn.leaky_relu(en_bnorm3,name=None,alpha = 0.15)
        
        
        flat_x = tf.contrib.layers.flatten(en_relu3) 
        
        fully_connected1 = tf.contrib.layers.fully_connected(inputs=flat_x, num_outputs=128, 
                                                         activation_fn=tf.nn.relu,scope="Fully_Conn1")
        fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=64, 
                                                         activation_fn=tf.nn.relu,scope="Fully_Conn2")
        fully_connected3 = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=32, 
                                                         activation_fn=tf.nn.relu,scope="Fully_Conn3")
        fully_connected3 = tf.contrib.layers.fully_connected(inputs=fully_connected3, num_outputs=16, 
                                                         activation_fn=tf.nn.relu,scope="Fully_Conn4")
        prediction = tf.contrib.layers.fully_connected(inputs=fully_connected3, num_outputs=2, 
                                               activation_fn=tf.nn.softmax,scope="Out")
        return prediction
        
        
    
    
def wce_loss(y,pred):
    #x_mask = tf.nn.softmax(x_mask,axis=2)
    lossv = tf.nn.weighted_cross_entropy_with_logits(
    y,
    pred,
    0.7,
    name=None
    )
    return tf.reduce_mean(lossv)

