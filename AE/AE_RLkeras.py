import numpy as np
import gym
import time
import json
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import TensorBoard
# seed = 417

# hyperparameters
Input_dim = 4*1 # input dimensionality
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

#set action label
action_up=2
action_down=3

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

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
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

#load json and create model
with open('Autoencoder_RL/data/Encoder_21_01.txt','r') as model_file:
    encoder = model_from_json(json.loads(next(model_file)))

# encoder_json_file = open('Autoencoder_RL/data/Encoder_21_01.txt','r')
# loaded_model_json = encoder_json_file.read()
# encoder_json_file.close()
# encoder = model_from_json(loaded_model_json)

encoder.load_weights('Autoencoder_RL/data/Encoder_21_01.h5')

encoder.summary()
if resume:
     # creates a generic neural network architecture
    dense_model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 2 units
    dense_model.add(Dense(units=2,input_dim=1*4,activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    dense_model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile the model using traditional Machine Learning losses and optimizers
    dense_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print model
    dense_model.summary()
    if os.path.isfile('AE_Rl_weights.h5'):
    #load pre-trained model weight
        print("loading previous weights")
        model.load_weights('AE_Rl_weights.h5')
else:
    # creates a generic neural network architecture
    dense_model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 2 units
    dense_model.add(Dense(units=2,input_dim=1*4,activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    dense_model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
    rms = optimizers.RMSprop(lr=learning_rate,decay=decay_rate)
    # compile the model using traditional Machine Learning losses and optimizers
    dense_model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

    #print model
    dense_model.summary()

    #save model
    # dense_model.save('dense_model.h5')

log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
callbacks = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None   # used in computing the difference frame

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_number = 0

#main loop for RL
while True:
    if render:env.render()
    #get the feature map from encoder output
    observation = prepro(observation)
    observation = np.array(observation)
    observation = observation.reshape(1,1,80,80)
    feature_map = encoder.predict(observation)
    feature_map = feature_map.flatten()
    # print(feature_map)
    # print("Previous",feature_map.shape)
    # feature_map = feature_map.flatten()
    # print(type(feature_map))
    # feature_map = feature_map.reshape(4,)
    # print(feature_map)
    # feature_map_shape = feature_map.get_shape()
    # print("After",feature_map.shape)

    if prev_x is not None:
      x = feature_map-prev_x
    else:
      x = np.zeros(Input_dim)
    prev_x = feature_map

    # print("x shape",x.shape)
    prob = dense_model.predict(np.expand_dims(x,axis=1).T)

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
        dense_model.fit(x=np.vstack(x_train),
                        y=np.vstack(y_train), 
                        verbose=1,
                        sample_weight=discount_rewards(rewards),
                        callbacks=[callbacks])
                        
        if episode_number % 100 == 0:
              model.save_weights('AE_Rl_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
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


