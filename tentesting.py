import os
import tentraining as tentraining
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
file = "3.jpg"
ori = img.imread(file)
ori = ori/255.0
image = tf.placeholder(tf.float32, shape=(1, ori.shape[0], ori.shape[1], ori.shape[2]))

out = tentraining.network(image)

saver = tf.train.Saver()
with tf.Session() as sess:
    
    if tf.train.get_checkpoint_state('./model_ten/'):  
        ckpt = tf.train.latest_checkpoint('./model_ten/')
        saver.restore(sess, ckpt)
        print ("load new model")


        
    detail_out  = sess.run(out, feed_dict={image:ori})
    
    detail_out = detail_out[0,:,:,:] 
    derained[np.where(derained < 0. )] = 0.
    derained[np.where(derained > 1. )] = 1.

    print(derained)
    plt.subplot(1,2,1)     
    plt.imshow(ori)      
    plt.title('input')   

    plt.subplot(1,2,2)    
    plt.imshow(derained)
    plt.title('output')   

    plt.show()
