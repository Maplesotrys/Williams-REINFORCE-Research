import numpy as np
import gym
import keras
import os
import json
import get_observation_samples as gos
from keras.models import model_from_json
from LossHistory import LossHistory
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import load_model
from keras.layers import Input,Dense, Flatten, Reshape,BatchNormalization, Activation, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from datetime import datetime
# from skimage.transform import resize
import keras.backend as K
from keras import callbacks
from keras.callbacks import TensorBoard

def preprocess(I):
  """ prepro 210x160x3 uint8 frame into 80x80 2D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)

env = gym.make('Pong-v0')


render = False
resume = False # checkpoint
prev_x = None
count = 0
input_dim = (80,80,1)
output_shape = (80,80)
padding = 'same'
d=20000
n=0
buffer=1000
k=0
observations=np.zeros((d,80,80)) 
# collect the sample observations
for i_episode in range(30):
    observation = env.reset()
    for t in range(10000):
        action = env.action_space.sample()
        # observations.append(observation)
        observation, reward, done, info = env.step(action)
        if n+t<d:
           observations[n+t]=preprocess(observation)
        else:
          pass
        #k+=1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # observations=np.array(observations)
            # print(observations.shape)
            break
    n+=t
    n+=1
    #if n>d:
       #break
# A= observations[1]
# b= observations[10000]
# c= observations[15000]
# print(A,A.shape)
# print(b,b.shape)
# print((A==b).all())
# print((b==c).all())
# print(observations)
# print('After shape',observations.shape)
observations = observations.reshape(20000,1,80,80)


if resume:
    with open('Autoencoder/data/Encoder.txt','r') as model_file:
        encoder = model_from_json(json.loads(next(model_file)))

    encoder.load_weights('Autoencoder/data/Encoder.h5')

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
    autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    autoencoder.summary()
    #autoencoder.save('Autoencoder_model.h5')
log_dir = './log_auto_encoder' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
callbacks = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,  
            write_graph=True, write_images=True)
#history = LossHistory()
autoencoder.fit(observations, observations,
                epochs=50,
                batch_size=32,
                verbose=1,
                shuffle=True,
                callbacks=[callbacks])

if not 'data' in os.listdir('.'):
    os.mkdir('./data/')
sample_features = encoder.predict(observations)
np.savez('./data/sample_features_20ksamples.npz', sample_features)

autoencoder.save_weights('./data/Autoencoder.h5')
autoencoder.to_json()

with open('./data/Autoencoder.txt', 'w') as outfile:
    json.dump(autoencoder.to_json(), outfile)

encoder.save_weights('./data/Encoder.h5')
encoder.to_json()

with open('./data/Encoder.txt', 'w') as outfile:
    json.dump(encoder.to_json(), outfile)


