import numpy as np
import gym
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import optimizers
from tf_log import tflog
from keras import callbacks
# seed = 417

# hyperparameters
Input_dim = 4*1 # input dimensionality
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

#set action label
action_up=2
action_down=3

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 80x80 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r) #need transfor to numpy array
  discounted_reward = np.zeros_like(r)  # initialize all elements as 0
  running_add = 0
  for t in reversed(range(len(discounted_reward))):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = r[t] + running_add * gamma  # the point here is to use Horner's method to compute those rewards efficiently
    discounted_reward[t] = running_add
  # standardize the rewards to be unit normal (helps control the gradient estimator variance)
  discounted_reward -= np.mean(discounted_reward) 
  discounted_reward /= np.std(discounted_reward)
  return discounted_reward


#load json and create model
with open('./data/Encoder.txt','r') as model_file:
    encoder = model_from_json(json.loads(next(model_file)))

# encoder_json_file = open('Autoencoder_RL/data/Encoder_21_01.txt','r')
# loaded_model_json = encoder_json_file.read()
# encoder_json_file.close()
# encoder = model_from_json(loaded_model_json)

encoder.load_weights('./data/Encoder.h5')

encoder.summary()
if resume:
     # creates a generic neural network architecture
    dense_model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 2 units
    #dense_model.add(Dense(units=2,input_dim=1*4,activation='relu', kernel_initializer='glorot_uniform'))
    dense_model.add(Dense(units=2,input_dim=4*1,activation='relu'))

    # output layer
    #dense_model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
    dense_model.add(Dense(units=2, activation='softmax'))
    #rms = optimizers.RMSprop(lr=learning_rate,decay=decay_rate)
    # compile the model using traditional Machine Learning losses and optimizers
    dense_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #print model
    dense_model.summary()
    if os.path.isfile('AE_Rl_weights.h5'):
    #load pre-trained model weight
        print("loading previous weights")
        dense_model.load_weights('AE_Rl_weights.h5')
else:
    # creates a generic neural network architecture
    dense_model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 2 units
    #dense_model.add(Dense(units=2,input_dim=1*4,activation='relu', kernel_initializer='glorot_uniform'))
    dense_model.add(Dense(units=2,input_dim=4*1,activation='relu'))

    # output layer
    #dense_model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
    dense_model.add(Dense(units=2, activation='softmax'))
    #rms = optimizers.RMSprop(lr=learning_rate,decay=decay_rate)
    # compile the model using traditional Machine Learning losses and optimizers
    dense_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #print model
    dense_model.summary()

    #save model
    # dense_model.save('dense_model.h5')

log_dir = './log_RL' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
callbacks = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)

env=gym.make("Pong-v0")
action_space=[2,3]
episodes = 0
n_episodes = 5000
reward_sums = np.zeros(n_episodes)
time_taken = np.zeros(n_episodes)
reward_sum = 0
reward_sum_tf=0
Input_dim = (4,)

prev_frame = None
Threshold = 1500
xs = np.zeros((Threshold,)+Input_dim)
ys = np.zeros((Threshold,1))
rs = np.zeros((Threshold))
run_re=[]
epoch =[]
k = 0
running_reward = None
observation = env.reset()
#main loop for RL
while episodes<n_episodes:
    if render:env.render()
    #get the feature map from encoder output

    observation = prepro(observation)
    observation = observation.reshape(1,1,80,80)
    feature_map = encoder.predict(observation)
    feature_map = feature_map.flatten()
    #print(feature_map.shape)
     # Get the current state of environment
    if prev_frame is not None :
      xs[k] = feature_map - prev_frame 
    else:
      np.zeros(Input_dim)
    prev_frame = feature_map
    
    
    # Take an action given current state of policy model
    p = dense_model.predict(np.expand_dims(xs[k],axis=1).T)
    a = np.random.choice(len(action_space), p=p[0])
    action = action_space[a]
    ys[k] = a
    
    # Renew state of environment
    observation, reward, done, _ = env.step(action)
    reward_sum += reward #record total rewards
    reward_sum_tf += reward
    rs[k] = reward # record reward per step
    
    k += 1
    
    if done or k==Threshold:
        print('At the end of episode', episodes, 'the total reward was :', reward_sum)
        reward_sums[episodes] = reward_sum
        reward_sum = 0
        
        # Gather state, action (y), and rewards (and preprocess)
        ep_x = xs[:k]
        ep_y = ys[:k]
        ep_r = rs[:k]
        ep_r = discount_rewards(ep_r)
        
        dense_model.fit(ep_x, ep_y, sample_weight=ep_r, batch_size=512, epochs=1, verbose=0,callbacks=[callbacks])
        # Log the reward
        running_reward = reward_sum_tf if running_reward is None else running_reward * 0.99 + reward_sum_tf * 0.01
        # if episode_number % 10 == 0:
        run_re.append(running_reward)
        epoch.append(episodes)
        tflog('running_reward', running_reward, custom_dir=log_dir)
        time_taken[episodes] = k
        k = 0
        prev_frame = None
        observation=env.reset()
        episodes += 1
        reward_sum_tf=0
        # save the model weight
        if episodes % 200 == 0:
            dense_model.save_weights('AE_RL_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
            ave_reward = np.mean(reward_sums[max(0,episodes-200):episodes])
            ave_time = np.mean(time_taken[max(0,episodes-200):episodes])
            print('Episode: {0:d},Average Reward: {1:.3f}, Average steps: {2:.3f}'.format(episodes,ave_reward, ave_time))
 

