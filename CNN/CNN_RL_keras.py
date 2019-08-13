# Conv Neural Network Keras version
import os
import numpy as np
import gym
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import optimizers
from datetime import datetime
import keras.backend as K
import matplotlib.pyplot as plt
from keras import callbacks
from keras.callbacks import TensorBoard
from tf_log import tflog

# hyperparameters
Input_dim = (80,80,1) # input dimensionality : 80x80 grid
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
epsilon = 1e-6

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
  return np.reshape(I,Input_dim)
#   return I.astype(np.float).ravel()
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
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r)+epsilon #idem
  return discounted_r

# def pgloss(y_true, y_pred):
#     """Policy Gradients loss. Maximizes log(output) · reward"""
#     return - K.mean(K.log(y_pred) * y_true)

if resume:
   # creates a generic neural network architecture
   model = Sequential()

   #conv1 output shape (32,80,80)
   model.add(Conv2D(filters=32, 
                    kernel_size=5, 
                    activation='relu',
                    padding ='same',
                    input_shape=Input_dim,
                    data_format='channels_last'))
   # pooling layer 1 output shape (32,40,40)
   model.add(MaxPooling2D(2))

   # conv2 output shape (64,40,40)
   model.add(Conv2D(64, 5 ,padding = 'same',activation='relu'))
  
   # pooling layer 2 output shape (64,20,20)
   model.add(MaxPooling2D(2))

   # conv3 output shape (64,20,20)
   model.add(Conv2D(64, 5, padding = 'same',activation='relu'))

   # pooling layer 3 output shape (64,10,10)
   model.add(MaxPooling2D(2))

   # fully connected layer 1 input shape(64*10*10)=6400, output shape (200)
   model.add((Flatten()))
   # model.add(Dense(units=200, activation='relu'))
   model.add(Dense(units=200, activation='relu', kernel_initializer='glorot_uniform'))
  
   # output layer
   # model.add(Dense(units=1, activation='sigmoid'))
   model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

   # compile the model using traditional Machine Learning losses and optimizers
   adam = optimizers.Adam(lr=learning_rate,decay=decay_rate)
   model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

   #print model
   model.summary()
   if os.path.isfile('CNN_Rl_weights.h5'):
    #load pre-trained model weight
        print("loading previous weights")
        model.load_weights('CNN_Rl_weights.h5')
else :
    # creates a generic neural network architecture
    model = Sequential()

    #conv1 output shape (32,80,80)
    model.add(Conv2D(filters=32, 
                     kernel_size=5, 
                     activation='relu',
                     padding ='same',
                     input_shape=Input_dim,
                     data_format='channels_last'))
    # pooling layer 1 output shape (32,40,40)
    model.add(MaxPooling2D(2))

    # conv2 output shape (64,40,40)
    model.add(Conv2D(64, 5 ,padding = 'same',activation='relu'))
   
    # pooling layer 2 output shape (64,20,20)
    model.add(MaxPooling2D(2))

    # conv3 output shape (64,20,20)
    model.add(Conv2D(64, 5, padding = 'same',activation='relu'))

    # pooling layer 3 output shape (64,10,10)
    model.add(MaxPooling2D(2))

    # fully connected layer 1 input shape(64*10*10)=6400, output shape (200)
    model.add((Flatten()))
    # model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=200, activation='relu', kernel_initializer='glorot_uniform'))
   
    # output layer
    # model.add(Dense(units=1, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile the model using traditional Machine Learning losses and optimizers
    adam = optimizers.Adam(lr=learning_rate,decay=decay_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    #print model
    model.summary()

    #save model
    # model.save('CNN_RL_model.h5')

log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
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
run_re=[]
epoch =[]

# for i_episode in range(10000):
    # main loop
while True:
    if render : env.render()
    # preprocess the observation, set input as difference between images
    cur_x = prepro(observation)
    # print(cur_x.shape)
    if prev_x is not None :
        x = cur_x - prev_x
    else:
        x = np.zeros(Input_dim)
    prev_x = cur_x

    # forward the policy network and sample action according to the proba distribution

    # two ways to calculate returned probability
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
        # print('At the end of episode', i_episode, 'the total reward was :', reward_sum)
        # dis_rewards=discount_rewards(rewards)
        # increment episode number
        episode_number += 1
        
        # training
        # model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards))
        model.fit(x=np.array(x_train),
                  y=np.array(y_train), 
                  verbose=1,
                  epochs=1,
                  sample_weight=discount_rewards(rewards),
                  callbacks=[callbacks])
        if episode_number % 100 == 0:
            model.save_weights('CNN_Rl_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
        # Log the reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        # if episode_number % 10 == 0:
        run_re.append(running_reward)
        epoch.append(episode_number)
        tflog('running_reward', running_reward, custom_dir=log_dir)
        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_x = None