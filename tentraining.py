import os
import tensorflow as tf
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from itertools import izip
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return np.array(images);

def network(inputs):
	with tf.variable_scope('conv_1'):
    		kernel = tf.Variable(tf.random_normal([3, 3, 3, 128], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights1')
    		biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases1')

    		conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
    		bias = tf.nn.bias_add(conv, biases)

    		conv1 = tf.nn.leaky_relu(bias)
	with tf.variable_scope('conv_2'):
    		kernel = tf.Variable(tf.random_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights2')
    		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases2')

    		conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='VALID')
    		bias = tf.nn.bias_add(conv, biases)

    		conv2 = tf.nn.leaky_relu(bias)
#	with tf.variable_scope('conv_3'):
#    		kernel = tf.Variable(tf.random_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights3')
#    		biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases3')

#    		conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='VALID')
#    		bias = tf.nn.bias_add(conv, biases)

#    		conv3 = tf.nn.leaky_relu(bias)
#	with tf.variable_scope('conv_4'):
#    		kernel = tf.Variable(tf.random_normal([3, 3, 256,512], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights4')
#    		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases4')

#    		conv = tf.nn.conv2d_transpose(conv3,kernel,[tf.shape(inputs)[0], tf.shape(inputs)[1]-4, tf.shape(inputs)[2]-4, 256],[1, 1, 1, 1],padding='VALID')   
#    		conv4 = tf.nn.bias_add(conv, biases)
#		conv4 = tf.nn.leaky_relu(conv4);
	with tf.variable_scope('conv_5'):
    		kernel = tf.Variable(tf.random_normal([3, 3, 128,256], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights5')
    		biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases5')

    		conv = tf.nn.conv2d_transpose(conv2,kernel,[tf.shape(inputs)[0], tf.shape(inputs)[1]-2, tf.shape(inputs)[2]-2, 128],[1, 1, 1, 1],padding='VALID')   
    		conv5 = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.leaky_relu(conv5)
	with tf.variable_scope('conv_6'):
    		kernel = tf.Variable(tf.random_normal([3, 3,1,128], dtype=tf.float32, stddev=1e-3), trainable=True, name='weights3')
    		biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), trainable=True, name='biases3')

    		conv = tf.nn.conv2d_transpose(tf.add(conv5,(0.9)*conv1),kernel,[tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], 1],[1, 1, 1, 1],padding='VALID')   
    		out = tf.nn.bias_add(conv, biases)
		out = tf.nn.relu(out);
	return out

if __name__ == '__main__':

  images = tf.placeholder(tf.float32, shape=(None, 128, 128, 3),name='images')
  labels = tf.placeholder(tf.float32, shape=(None, 128, 128, 1),name='labels')
  
  outputs = network(images)
  
  loss = tf.reduce_mean(tf.square(labels - outputs))  # MSE loss
  
  lr_ = 1e-3
  lr  = tf.placeholder(tf.float32 ,shape = []) 
  
  g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam

  saver = tf.train.Saver(max_to_keep = 5)

  
  
  epoch = int(30) 
  datagen = tf.keras.preprocessing.image.ImageDataGenerator();
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    

    #validation_data_name = "validation.h5"
    #validation_data, validation_label = read_data(data_path + validation_data_name)

    #validation_data = np.transpose(validation_data, (0,2,3,1))
   

    
#validation_label = np.transpose(validation_label, (0,2,3,1))
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = "model_ten/";
    if tf.train.get_checkpoint_state('./model_ten/'):   # load previous trained model 
      ckpt = tf.train.latest_checkpoint('./model_ten/')
      saver.restore(sess, ckpt)
      ckpt_num = re.findall(r"\d",ckpt)
      if len(ckpt_num) == 3:
        start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
      elif len(ckpt_num) == 2:
        start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
      else:
        start_point = int(ckpt_num[0])      
      print("Load success")
   
    else:  # re-training when no model found
      print("re-training")
      start_point = 0   

    val_data = load_images_from_folder('val/');
    val_labels=load_images_from_folder('val_labels/');
    print("val_data_shape: ",val_data.shape);
    print("val_labels_shape: ",val_labels.shape);
    for j in range(start_point,epoch):   # epoch
      training_loss=0;
      for data,labeled in izip(datagen.flow_from_directory('resizedImg/',target_size=(128,128),batch_size=5),datagen.flow_from_directory('bw/',target_size=(128,128),batch_size=5,color_mode="grayscale")):
 	print(np.array(data[0]).shape);
	print(np.array(labeled[0]).shape);
        data1 = np.array(data[0]).astype(np.float32)/255.0;
	labels1= np.array(labeled[0]).astype(np.float32)/255.0;
        
	_,train_loss = sess.run([g_optim,loss], feed_dict={images:data1, labels:labels1, lr: lr_});
        training_loss+=train_loss/64;
	print("epoch: "+str(epoch))
      Validation_Loss  = sess.run(loss,  feed_dict={images: val_data, labels:val_labels}) # validation loss 
      model_name = 'model-epoch.h5'  # save model
      save_path_full = os.path.join(save_path, model_name)
      saver.save(sess, save_path_full, global_step = j+1)
      print ('%d epoch is finished, Training_Loss = %.4f, Validation_Loss = %.4f' %
               (j+1, Training_Loss, Validation_Loss))
