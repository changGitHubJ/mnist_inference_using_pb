import numpy as np
import tensorflow as tf

class Model:
    IMG_SIZE = 0
    # TEST_DATA_SIZE = 0
    learning_rate = 0

    def __init__(self, img_size, learning_rate):
        self.IMG_SIZE = img_size
        # self.TEST_DATA_SIZE = test_size
        self.learning_rate = learning_rate

    def conv2d(self, input, weight_shape, bias_shape, layer_name):
        incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
        W = tf.get_variable("W_" + layer_name, weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value=0)
        b = tf.get_variable("b_" + layer_name, bias_shape, initializer=bias_init)
        return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

    def max_pool(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def layer(self, input, weight_shape, bias_shape, layer_name):
        weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W_" + layer_name, weight_shape, initializer=weight_init)
        b = tf.get_variable("b_" + layer_name, bias_shape, initializer=bias_init)
        return tf.nn.sigmoid(tf.matmul(input, W) + b)

    def output_layer(self, input, weight_shape, bias_shape, layer_name):
        weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W_" + layer_name, weight_shape, initializer=weight_init)
        b = tf.get_variable("b_" + layer_name, bias_shape, initializer=bias_init)
        return tf.nn.softmax(tf.matmul(input, W) + b, name='y')

    def inference(self, x, keep_prob):
        x = tf.reshape(x, shape=[-1, self.IMG_SIZE, self.IMG_SIZE, 1])
        conv_1 = self.conv2d(x, [5, 5, 1, 32], [32], 'conv_1')
        pool_1 = self.max_pool(conv_1)
        conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64], 'conv_2')
        pool_2 = self.max_pool(conv_2)
        pool_2_flat = tf.reshape(pool_2, [-1, 7*7* 64])
        fc_1 = self.layer(pool_2_flat, [7*7*64, 1024], [1024], 'fc')
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
        output = self.output_layer(fc_1_drop, [1024, 10], [10], 'output')
        
        return output

    def loss(self, output, weight):

        soft = tf.nn.softmax(output)
        xentropy = - tf.reduce_sum(weight * tf.log(soft), 1)
        loss = tf.reduce_mean(xentropy)
        return loss

    def training(self, cost, global_step):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op

    def evaluate(self, output, y):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar("validation error", (1.0 - accuracy))
        return accuracy