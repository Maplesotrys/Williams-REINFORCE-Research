import numpy as np
# import cPickle as pickle
import matplotlib.pyplot as plt
#from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import gym
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import rmsprop
import keras.backend as K
from keras import callbacks
from keras.callbacks import TensorBoard
from tf_log import tflog
env=gym.make("Pong-v0")
action_space=[2,3]
gamma = 0.99
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(discounted_r))):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add =  r[t] + running_add * gamma # belman equation
        discounted_r[t] = running_add
    return discounted_r

def discount_n_standardise(r):
    dr = discount_rewards(r)
    dr = (dr - dr.mean()) / dr.std()
    return dr

def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)[:,:,None]

model = Sequential()
model.add(Conv2D(4, kernel_size=(3,3), padding='same', activation='relu', input_shape = (80,80,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(len(action_space), activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['accuracy']) #

model.summary()

log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
callbacks = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)

episodes = 0
n_episodes = 5000
reward_sums = np.zeros(n_episodes)
losses = np.zeros(n_episodes)
time_taken = np.zeros(n_episodes)
reward_sum = 0
reward_sum_tf=0
im_shape = (80, 80, 1)

prev_frame = None
buffer = 1000
xs = np.zeros((buffer,)+im_shape)
ys = np.zeros((buffer,1))
rs = np.zeros((buffer))
run_re=[]
epoch =[]
k = 0
running_reward = None
observation = env.reset()

while episodes<n_episodes:
    # Get the current state of environment
    x = preprocess(observation)
    xs[k] = x - prev_frame if prev_frame is not None else np.zeros(im_shape)
    prev_frame = x
    
    # Take an action given current state of policy model
    p = model.predict(xs[k][None,:,:,:])
    a = np.random.choice(len(action_space), p=p[0])
    action = action_space[a]
    ys[k] = a
    
    # Renew state of environment
    observation, reward, done, _ = env.step(action)
    reward_sum += reward #record total rewards
    reward_sum_tf += reward
    rs[k] = reward # record reward per step
    
    k += 1
    
    if done or k==buffer:
        print('At the end of episode', episodes, 'the total reward was :', reward_sum_tf)
        reward_sums[episodes] = reward_sum
        reward_sum = 0
        
        # Gather state, action (y), and rewards (and preprocess)
        ep_x = xs[:k]
        ep_y = ys[:k]
        ep_r = rs[:k]
        ep_r = discount_n_standardise(ep_r)
        
        model.fit(ep_x, ep_y, sample_weight=ep_r, batch_size=512, epochs=1, verbose=0,callbacks=[callbacks])
        # Log the reward
        running_reward = reward_sum_tf if running_reward is None else running_reward * 0.99 + reward_sum_tf * 0.01
        # if episode_number % 10 == 0:
        run_re.append(running_reward)
        epoch.append(episodes)
        tflog('running_reward', running_reward, custom_dir=log_dir)
        time_taken[episodes] = k
        k = 0
        prev_frame = None
        observation = env.reset()
        '''losses[episodes] = model.evaluate(ep_x, 
                                          ep_y,
                                          sample_weight=ep_r,
                                          batch_size=len(ep_x), 
                                          verbose=0)'''
        episodes += 1
        reward_sum_tf=0
        if episodes % 500 == 0:
            model.save_weights('CNN_RL_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
        # Log the reward
        # Print out metrics like rewards, how long each episode lasted etc.
        if episodes%(n_episodes//20) == 0:
            ave_reward = np.mean(reward_sums[max(0,episodes-200):episodes])
            ave_loss = np.mean(losses[max(0,episodes-200):episodes])
            ave_time = np.mean(time_taken[max(0,episodes-200):episodes])
            print('Episode: {0:d}, Average Loss: {1:.4f}, Average Reward: {2:.4f}, Average steps: {3:.4f}'
                  .format(episodes, ave_loss, ave_reward, ave_time))
