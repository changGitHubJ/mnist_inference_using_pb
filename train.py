import numpy as np
import os
import random
import tensorflow as tf
import time

import model

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameter
learning_rate = 0.0001
training_epochs = 100
batch_size = 100
display_step = 1
TRAIN_DATA_SIZE = 10000
VALID_DATA_SIZE = 500
TEST_DATA_SIZE = 500
IMG_SIZE = 28
OUTPUT_SIZE = 10
FILTER_SIZE_1 = 32
FILTER_SIZE_2 = 64

# Batch components
trainingImages = np.zeros((TRAIN_DATA_SIZE, IMG_SIZE*IMG_SIZE + 1))
trainingLabels = np.zeros((TRAIN_DATA_SIZE, OUTPUT_SIZE + 1))
trainingWeights = np.zeros((TRAIN_DATA_SIZE, OUTPUT_SIZE + 1))
validationImages = np.zeros((VALID_DATA_SIZE, IMG_SIZE*IMG_SIZE))
validationLabels = np.zeros((VALID_DATA_SIZE, OUTPUT_SIZE))
validationWeights = np.zeros((VALID_DATA_SIZE, OUTPUT_SIZE))
testImages = np.zeros((TEST_DATA_SIZE, IMG_SIZE*IMG_SIZE))
testLabels = np.zeros((TEST_DATA_SIZE, OUTPUT_SIZE))
testWeights = np.zeros((TEST_DATA_SIZE, OUTPUT_SIZE))

def write_result(output, y, model):
    return output, y, model

def openfile(filename):
    file = open(filename)
    VAL = []
    while True:
        line = file.readline()
        if(not line):
            break
        val = line.split(' ')
        VAL.append(val)
    return VAL

def read_training_data():
    fileImg = open('./data/trainImage.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        trainingImages[i, :] = val
    for i in range(TRAIN_DATA_SIZE):
        for j in range(1,IMG_SIZE*IMG_SIZE + 1):
            trainingImages[i, j] /= 255.0

    filelbl = open('./data/trainLABEL.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        trainingLabels[i, :] = val
    
    filewgh = open('./data/trainWEIGHT.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        trainingWeights[i, :] = val

def defineBatchComtents():
    num = np.linspace(0, TRAIN_DATA_SIZE - 1, TRAIN_DATA_SIZE)
    num = num.tolist()
    COMPONENT = []
    total_batch = int(TRAIN_DATA_SIZE/batch_size)
    for i in range(total_batch):
        component = random.sample(num, batch_size)
        COMPONENT.append(component)
        for j in range(batch_size):
            cnt = 0
            while True:
                if(num[cnt] == component[j]):
                    num.pop(cnt)
                    break
                else:
                    cnt += 1
    
    return COMPONENT

def next_batch(batch_component):
    num = sorted(batch_component)
    lineNum = 0
    cnt = 0
    batch_x = []
    batch_y = []
    batch_weight = []
    while True:
        if(cnt == batch_size):
            break
        else:
            if(int(num[cnt]) == int(trainingImages[lineNum, 0])):
                image = trainingImages[lineNum, 1:IMG_SIZE*IMG_SIZE + 1]
                label = trainingLabels[lineNum, 1:OUTPUT_SIZE + 1]
                weight = trainingWeights[lineNum, 1:OUTPUT_SIZE + 1]
                batch_x.append(image)
                batch_y.append(label)
                batch_weight.append(weight)
                cnt += 1
        lineNum += 1

    return np.array(batch_x), np.array(batch_y), np.array(batch_weight)

def read_validation_data():
    fileImg = open('./data/validationImage.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        validationImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(VALID_DATA_SIZE):
        for j in range(IMG_SIZE*IMG_SIZE):
            validationImages[i, j] /= 255.0

    filelbl = open('./data/validationLABEL.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        validationLabels[i, :] = val[1:OUTPUT_SIZE + 1]
    
    filewgh = open('./data/validationWEIGHT.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        validationWeights[i, :] = val[1:OUTPUT_SIZE + 1]

def read_test_data():
    fileImg = open('./data/testImage.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        testImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(TEST_DATA_SIZE):
        for j in range(IMG_SIZE*IMG_SIZE):
            testImages[i, j] /= 255.0
    
    filelbl = open('./data/testLABEL.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        testLabels[i, :] = val[1:OUTPUT_SIZE + 1]

    filewgh = open('./data/testWEIGHT.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        testWeights[i, :] = val[1:OUTPUT_SIZE + 1]

def write_log(msg):
    os.makedirs('./log', exist_ok=True)
    file = open('./log/trainingLog.txt', mode='a')
    file.write(msg + '\n')
    file.close()

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            x = tf.placeholder("float", [None, IMG_SIZE*IMG_SIZE], name='x')
            weight = tf.placeholder("float", [None, OUTPUT_SIZE], name='y_')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            read_training_data()
            read_validation_data()
            read_test_data()
            
            model = model.Model(IMG_SIZE, learning_rate)
            output = model.inference(x, keep_prob)
            cost = model.loss(output, weight)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = model.training(cost, global_step)
            eval_op = model.evaluate(output, weight)
            saver = tf.train.Saver()
            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(TRAIN_DATA_SIZE/batch_size)
                batch_component = defineBatchComtents()
                # loop over all batchs
                for i in range(total_batch):
                    minibatch_x, minibatch_y, minibatch_weight = next_batch(batch_component[i])
                    sess.run(train_op, feed_dict={x: minibatch_x, weight: minibatch_weight, keep_prob: 0.5})
                    avg_cost += sess.run(cost, feed_dict={x: minibatch_x, weight: minibatch_weight, keep_prob: 0.5})/total_batch

                # display logs per step
                if epoch % display_step == 0:
                    accuracy = sess.run(eval_op, feed_dict={x: validationImages, weight: validationWeights, keep_prob: 0.5})
                    msg = "Epoch: " + str(epoch+1) + ", cost = " + "{:.9f}".format(avg_cost) + ", Validation Error = " + "{:.9f}".format(1 - accuracy)
                    print(msg)
                    write_log(msg)

            print("Optimizer finished!")
            accuracy = sess.run(eval_op, feed_dict={x: testImages, weight: testWeights, keep_prob: 1})
            msg = "Epoch: " + str(-1) + ", cost = " + "{:.9f}".format(-1) + ", Test Accuracy = " + "{:.9f}".format(accuracy)
            print(msg)
            write_log(msg)
            
            # output model
            saver.save(sess, "./model")

            print('Saved a model.')

            sess.close()
                

