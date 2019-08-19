import numpy as np
import gym
import keras
import os
import json
from keras.models import model_from_json
from LossHistory import LossHistory
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import load_model
from keras.layers import Input,Dense, Flatten, Reshape,BatchNormalization, Activation, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
# from skimage.transform import resize
import keras.backend as K

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

env = gym.make('Pong-v0')

max_observations = 20000
observations = []
render = False
resume = False # checkpoint
prev_x = None
count = 0
input_dim = (80,80,1)
output_shape = (80,80)
padding = 'same'

while True:
    if len(observations)>=max_observations : 
      break
    observation = env.reset()
    if count % 10 == 0:
        # preprocess the observation, set input as difference between images
        x = prepro(observation)
        observations.append(x)
    count +=1
    done = False

    while not done:
        if render: env.render()
        if len(observations) >= max_observations:
          break
        a = env.action_space.sample()
        obs,r,done,info = env.step(a)
        if count % 10 ==0:
            x = prepro(observation)
            observations.append(x)
            if not len(observations) % 1000:
                print(len(observations))
        a = env.action_space.sample()
        count +=1
# observations = np.array(observations)
# print(observations.shape)
env.close()

observations = np.array(observations)
# print('Previous shape',observations.shape)
observations = observations.reshape(20000,1,80,80)
print(observations)
# print('After shape',observations.shape)


if resume:
    with open('Autoencoder_RL/data/Encoder_21_01.txt','r') as model_file:
        encoder = model_from_json(json.loads(next(model_file)))

    encoder.load_weights('Autoencoder_RL/data/Encoder_21_01.h5')

    encoder.summary()
    encoder_output = encoder.predict(observations)
    print(encoder_output)
    print(encoder_output.shape)
    print(type(encoder_output))
else:
    input_img = Input(shape=observations.shape[1:])
    # print(input_img.shape)
    x = Conv2D(filters=16, kernel_size=5, activation='relu',input_shape=input_dim, padding=padding,data_format='channels_first')(input_img)
    x = MaxPooling2D((2, 2),data_format='channels_first', padding=padding)(x)
    x = Conv2D(32, 5, activation='relu', padding=padding,data_format='channels_first')(x)
    x = MaxPooling2D((2, 2),data_format='channels_first', padding=padding)(x)
    # filters_shape = x.get_shape()
    flattened = Flatten()(x)
    # flat_shape = flattened.get_shape()
    encoded = Dense(4, activation='relu')(flattened)

    # x = Dense(units=int(flat_shape[1]), activation='relu')(encoded)
    x = Dense(units=12800,activation='relu')(encoded)
    x = Reshape((32,20,20))(x)
    x = Conv2D(32, 5,  activation='relu', padding=padding,data_format='channels_first')(x)
    x = UpSampling2D((2, 2),data_format='channels_first')(x)
    x = Conv2D(16, 5,  activation='relu', padding=padding,data_format='channels_first')(x)
    x = UpSampling2D((2, 2),data_format='channels_first')(x)
    x = Conv2D(1, 5,  activation='relu', padding=padding,data_format='channels_first')(x)

    decoded = x

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    autoencoder.save('Autoencoder_model.h5')

history = LossHistory()
autoencoder.fit(observations, observations,
                epochs=10,
                batch_size=32,
                verbose=1,
                shuffle=True,
                callbacks=[history])

if not 'data' in os.listdir('Autoencoder_RL'):
    os.mkdir('Autoencoder_RL/data/')
sample_features = encoder.predict(observations)
np.savez('Autoencoder_RL/data/sample_features_20k.npz', sample_features)

autoencoder.save_weights('Autoencoder_RL/data/Autoencoder_21_01.h5')
autoencoder.to_json()

with open('Autoencoder_RL/data/Autoencoder_21_01.txt', 'w') as outfile:
    json.dump(autoencoder.to_json(), outfile)

encoder.save_weights('Autoencoder_RL/data/Encoder_21_01.h5')
encoder.to_json()

with open('Autoencoder_RL/data/Encoder_21_01.txt', 'w') as outfile:
    json.dump(encoder.to_json(), outfile)


