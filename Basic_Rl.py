# Dense Neural Network Keras version

import numpy as np
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# from LossHistory import LossHistory
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from keras import callbacks
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpu_options = tf.GPUOptions(allow_growth=True)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# KTF.set_session(sess)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #设置需要使用的GPU的编号
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4 #设置使用GPU容量占GPU总容量的比例
# sess = tf.Session(config=config)
# KTF.set_session(sess)
# hyperparameters
Input_dim = 80 * 80 # input dimensionality : 80x80 grid
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

#set action label
action_up=2
action_down=3

# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r) #need transfor to numpy array
  discounted_r = np.zeros_like(r)  # initialize all elements as 0
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
    
  # standardize the rewards to be unit normal (helps control the gradient estimator variance)
  discounted_r -= np.mean(discounted_r) 
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

if resume:
    # creates a generic neural network architecture
    model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 200 units
    model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile the model using traditional Machine Learning losses and optimizers
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print model
    model.summary()

    #load pre-trained model weight
    model.load_weights('my_model_weights.h5')

else :
    # creates a generic neural network architecture
    model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 200 units
    model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile the model using traditional Machine Learning losses and optimizers
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print model
    model.summary()

    #save model
    model.save_weights('my_model_weights.h5')

callbacks = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None   # used in computing the difference frame

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_number = 0


# main loop
while True:
    if render : env.render()
    # preprocess the observation, set input as difference between images
    cur_x = prepro(observation)
    # i=np.expand_dims(cur_x,axis=0)
    # print(i.shape)
    # print(cur_x.shape)
    if prev_x is not None :
        x = cur_x - prev_x
    else:
        x = np.zeros(Input_dim)
    # print(x.shape)
    # print(np.expand_dims(cur_x,axis=0).shape)
    prev_x = cur_x
    
    # forward the policy network and sample action according to the proba distribution

    # two ways to calculate returned probability
    # print(x.shape)
    prob = model.predict(np.expand_dims(x, axis=0))
    # aprob = model.predict(np.expand_dims(x, axis=1).T)
    
    if np.random.uniform() < prob:
        action = action_up
    else :
        action = action_down

    # 0 and 1 labels( a fake label in order to achive back propagation algorithm)
    if action == 2:
        y = 1     
    else:
        y = 0 

    # log the input and label to train later
    x_train.append(x)
    y_train.append(y)

    # do one step in our environment
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward
    
    # end of an episode
    if done:
        print('At the end of episode', episode_number, 'the total reward was :', reward_sum)
        
        # increment episode number
        episode_number += 1
        
        # training
        # history = LossHistory()
        model.fit(x=np.vstack(x_train), 
                  y=np.vstack(y_train), 
                  verbose=1, 
                  sample_weight=discount_rewards(rewards))
                #   callbacks=[TensorBoard(log_dir='mytensorboard')])

                                                    
        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_x = None
        
