import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image

import model

IMG_SIZE = 28
OUTPUT_SIZE = 10
DATA_SIZE = 500
MAXIMUM_ANALYSIS_SIZE = 30

dataImages = np.zeros((DATA_SIZE, IMG_SIZE*IMG_SIZE))

def read_data():
    fileImg = open('./data/testImage.txt', 'r')
    for i in range(DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        dataImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    
if __name__=='__main__':    
    # with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE]) # inputs(gray image)

            read_data()

            model = model.Model(IMG_SIZE, 0)
            output = model.inference(x, 1.0)
            saver = tf.train.Saver()
            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            model_path = tf.train.latest_checkpoint("./")
            saver.restore(sess, model_path)

            if not os.path.exists('./result'):
                os.mkdir('./result')
            remaining_size = DATA_SIZE
            ANALYSIS_SIZE = MAXIMUM_ANALYSIS_SIZE
            start = 0
            counter = 0
            while True:
                if(remaining_size > 0 and remaining_size - MAXIMUM_ANALYSIS_SIZE >= 0):
                    ANALYSIS_SIZE = MAXIMUM_ANALYSIS_SIZE
                elif(remaining_size > 0 and remaining_size - MAXIMUM_ANALYSIS_SIZE < 0):
                    ANALYSIS_SIZE = remaining_size
                else:
                    break

                decoded_imgs = sess.run(output, feed_dict={x: dataImages[start:start + ANALYSIS_SIZE,:]})
                decoded_imgs = decoded_imgs.reshape([-1, OUTPUT_SIZE])
                
                for i in range(ANALYSIS_SIZE):
                    filename = './result/data' + str(counter) + '.txt'
                    print('writing ' + filename)
                    np.savetxt(filename, decoded_imgs[i, :])
                    counter += 1

                remaining_size -= ANALYSIS_SIZE 
                start = start + ANALYSIS_SIZE

            for k in range(DATA_SIZE):
                plt.figure(figsize=(7, 4))
                plt.subplot(1, 2, 1)
                fig = plt.imshow(dataImages[k, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='gray')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)

                output = np.loadtxt('./result/data' + str(k) + '.txt')
                plt.subplot(1, 2, 2)
                n = np.linspace(0, 9, 10)
                fig = plt.plot(output.reshape([OUTPUT_SIZE, 1]), n)
                plt.show()