################################################################################################
########################################## INITIALIZE ##########################################
################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from scipy.io import loadmat
from time import time

from sklearn.metrics import mean_squared_error
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity

import keras
import keras.backend as K
from keras import Model, Input
from tensorflow_addons.layers import InstanceNormalization, GELU
from keras.layers import BatchNormalization, LayerNormalization, Dropout, PReLU
from keras.layers import Flatten, Reshape, Concatenate, Lambda
from keras.layers import SeparableConv2D, AveragePooling2D, UpSampling2D, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss_mse

################################################################################################
########################################## DATALOADER ##########################################
################################################################################################
n_realizations = 1000
n_timesteps    = 45
xy_dim         = 128

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

def my_normalize(data, original_feature, mode):
    feature_min = original_feature.min(axis=(-2,-1), keepdims=True)
    feature_max = original_feature.max(axis=(-2,-1), keepdims=True)
    if mode=='forward':
        return (data-feature_min)/(feature_max-feature_min)
    elif mode=='inverse':
        return data*(feature_max-feature_min)+feature_min

def make_initial_data():
    well_opr   = np.zeros((n_realizations, n_timesteps, 3))
    well_wpr   = np.zeros((n_realizations, n_timesteps, 3))
    well_wcut  = np.zeros((n_realizations, n_timesteps, 3))
    perm       = np.zeros(shape=(n_realizations, xy_dim, xy_dim))
    poro       = np.zeros(shape=(n_realizations, xy_dim, xy_dim))
    pressure   = np.zeros(shape=(n_realizations, n_timesteps, xy_dim, xy_dim))
    saturation = np.zeros(shape=(n_realizations, n_timesteps, xy_dim, xy_dim))
    timestamps = loadmat('simulations_2D/response_production/production_1.mat')['Prod']['t'][0][0].flatten()[1:]*3.1689E-8
    channels = np.transpose(np.array(pd.read_csv('simulations_2D/channel_all.csv', header=None)).T.reshape(n_realizations, xy_dim,xy_dim), axes=(0,2,1))
    for i in range(n_realizations):
        well_opr[i] = loadmat('simulations_2D/response_production/production_{}.mat'.format(i+1))['Prod']['opr'][0][0][1:, 3:]*5.4344E5 #to bbls
        well_wpr[i] = loadmat('simulations_2D/response_production/production_{}.mat'.format(i+1))['Prod']['wpr'][0][0][1:, 3:]*5.4344E5 #to bbls
        well_wcut[i] = loadmat('simulations_2D/response_production/production_{}.mat'.format(i+1))['Prod']['wc'][0][0][1:, 3:]
    for i in range(n_realizations):
        poro[i,:,:] = loadmat('simulations_2D/features_porosity/porosity_{}.mat'.format(i+1))['poro'].flatten().reshape(xy_dim,xy_dim)
        perm[i,:,:] = np.log10(loadmat('simulations_2D/features_permeability/permeability_{}.mat'.format(i+1))['permeability']).flatten().reshape(xy_dim,xy_dim)
        pressure[i,:,:,:]   = loadmat('simulations_2D/response_pressure/pressure_{}.mat'.format(i+1))['pres'].T.reshape(n_timesteps,xy_dim,xy_dim)**0.00689476 #to psi
        saturation[i,:,:,:] = loadmat('simulations_2D/response_saturation/saturation_{}.mat'.format(i+1))['satu'].T.reshape(n_timesteps,xy_dim,xy_dim)
    np.save('simulations_2D/data/well_opr.npy', well_opr)
    np.save('simulations_2D/data/well_wpr.npy', well_wpr)
    np.save('simulations_2D/data/well_wcut.npy', well_wcut)
    np.save('simulations_2D/data/poro.npy', poro)
    np.save('simulations_2D/data/perm.npy', perm)
    np.save('simulations_2D/data/channels.npy', channels)
    np.save('simulations_2D/data/pressure.npy', pressure)
    np.save('simulations_2D/data/saturation.npy', saturation)
    np.save('simulations_2D/data/timestamps.npy', timestamps)
    return timestamps, poro, perm, channels, pressure, saturation, well_opr, well_wpr, well_wcut

def load_initial_data():
    timestamps = np.load('simulations_2D/data/timestamps.npy')
    poro       = np.load('simulations_2D/data/poro.npy')
    perm       = np.load('simulations_2D/data/perm.npy')
    channels   = np.load('simulations_2D/data/channels.npy')
    pressure   = np.load('simulations_2D/data/pressure.npy')
    saturation = np.load('simulations_2D/data/saturation.npy')
    well_opr   = np.load('simulations_2D/data/well_opr.npy')
    well_wpr   = np.load('simulations_2D/data/well_wpr.npy')
    well_wcut  = np.load('simulations_2D/data/well_wcut.npy')
    print('Perm: {} | Poro: {} | Channels: {} | Pressure: {} | Saturation: {}'.format(perm.shape, poro.shape, channels.shape, pressure.shape, saturation.shape))
    print('OPR: {} | WPR: {} | WCUT: {} | Timestamps: {}'.format(well_opr.shape, well_wpr.shape, well_wcut.shape, timestamps.shape))
    return timestamps, poro, perm, channels, pressure, saturation, well_opr, well_wpr, well_wcut

def split_xyw(poro, perm, channels, pressure, saturation, well_opr, well_wpr, well_wcut):
    X_data = np.concatenate((np.expand_dims(my_normalize(pressure, pressure, 'forward'), -1),
                             np.expand_dims(my_normalize(saturation, saturation, 'forward'), -1)), -1)
    y_data = np.concatenate((np.expand_dims(my_normalize(poro, poro, 'forward'), -1),
                             np.expand_dims(my_normalize(perm, perm, 'forward'), -1),
                             np.expand_dims(channels, -1)), -1)
    w_data = np.concatenate((np.expand_dims(my_normalize(well_opr[:,1:], well_opr[:,1:], 'forward'), -1),
                             np.expand_dims(my_normalize(well_wpr[:,1:], well_wpr[:,1:], 'forward'), -1),
                             np.expand_dims(my_normalize(well_wcut[:,1:], well_wcut[:,1:], 'forward'), -1)), -1)
    np.save('data/X_data.npy', X_data); np.save('data/y_data.npy', y_data); np.save('data/w_data.npy', w_data)
    print('X shape: {} | w shape: {} | y shape: {}'.format(X_data.shape, w_data.shape, y_data.shape))
    return X_data, y_data, w_data

def load_xywt():
    x = np.load('simulations_2D/data/X_data.npy')
    y = np.load('simulations_2D/data/y_data.npy')
    w = np.load('simulations_2D/data/w_data.npy')
    t = np.load('simulations_2D/data/timestamps.npy')
    print('X shape: {} | w shape: {} \ny shape: {} | t shape: {}'.format(x.shape, w.shape, y.shape, t.shape))
    return x, y, w, t

def my_train_test_split(X, y, w, nobs, split_perc=0.7, n_realizations=n_realizations, xy_dim=xy_dim):
    train_size = int(np.ceil(n_realizations*split_perc))
    randx, randy = [np.random.randint(xy_dim, size=nobs) for _ in range(2)]
    X_train, X_test = X[:train_size,:,randx,randy], X[train_size:,:,randx,randy]
    w_train, w_test = w[:train_size], w[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print('X_train shape: {}   | X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('w_train shape: {}    | w_test shape: {}'.format(w_train.shape, w_test.shape))
    print('y_train shape: {} | y_test shape: {}'.format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, w_train, w_test, randx, randy

def make_inv_backnorm(inv_tuple, true_tuple):
    inv_train, inv_test = inv_tuple
    inv_all = np.concatenate([inv_train, inv_test])
    poro, perm, channels = true_tuple
    poro_hat = my_normalize(inv_all[...,0], poro, mode='inverse')
    perm_hat = my_normalize(inv_all[...,1], perm, mode='inverse')
    chan_hat = my_normalize(inv_all[...,2], channels, mode='inverse')
    return poro_hat, perm_hat, chan_hat

################################################################################################
########################################## PLOT UTILS ##########################################
################################################################################################
def plot_loss(fit, title='', figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    loss = fit.history['loss']
    val  = fit.history['val_loss']
    epochs = len(loss)
    iterations = np.arange(epochs)
    plt.plot(iterations, loss, '-', label='loss')
    plt.plot(iterations, val, '-', label='validation loss')
    plt.title(title+' Training: Loss vs epochs'); plt.legend()
    plt.ylabel('Epochs'); plt.ylabel('Loss')
    plt.xticks(iterations[::epochs//10])

def plot_loss_all(loss1, loss2, loss3, title1='Data', title2='Static', title3='Dynamic', figsize=(15,3)):
    plt.figure(figsize=figsize, facecolor='white')
    plt.subplot(131); plot_loss(loss1, title=title1)
    plt.subplot(132); plot_loss(loss2, title=title2)
    plt.subplot(133); plot_loss(loss3, title=title3)

def plot_static(poro, perm, channels, multiplier=1, ncols=10, figsize=(20,4), cmaps=['binary', 'viridis', 'jet']):
    labels = ['facies', 'porosity', 'permeability']
    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=figsize, facecolor='white')
    for i in range(10):
        k = i*multiplier
        im1 = axs[0,i].imshow(channels[k], cmap=cmaps[0])
        im2 = axs[1,i].imshow(poro[k], cmap=cmaps[1])
        im3 = axs[2,i].imshow(perm[k], cmap=cmaps[2])
        axs[0,i].set(title='Realization {}'.format(k))
        plt.colorbar(im1, ticks=range(2)); plt.colorbar(im2); plt.colorbar(im3)
        for j in range(3):
            axs[j,i].set(xticks=[], yticks=[])
            axs[j,0].set(ylabel=labels[j])

def plot_dynamic(static, dynamic, multiplier=1, nrows=5, figsize=(30,8), cmaps=['jet', 'gnuplot2']):
    fig, axs = plt.subplots(nrows=nrows, ncols=16, figsize=figsize, facecolor='white')
    for i in range(nrows):
        k = i*multiplier
        im1 = axs[i,0].imshow(static[k], cmap=cmaps[0])
        axs[i,0].set(ylabel='realization {}'.format(k))
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        for j in range(15):
            im2 = axs[i,j+1].imshow(dynamic[k,j*3], cmap=cmaps[1])
            axs[0,j+1].set(title='state {}'.format(j*3))
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        for j in range(16):
            axs[i,j].set(xticks=[], yticks=[])

def plot_data(timestamps, opr, wpr, wcut, multiplier=1, ncols=10, figsize=(20,6)):
    labels     = ['Oil Rate [bpd]', 'Water Rate [bpd]', 'Water Cut [%]']
    well_names = ['P1', 'P2', 'P3']
    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=figsize, facecolor='white')
    for i in range(ncols):
        axs[0,i].plot(timestamps, opr[i*multiplier])
        axs[1,i].plot(timestamps, wpr[i*multiplier])
        axs[2,i].plot(timestamps, wcut[i*multiplier])
        axs[0,i].set_title('realization {}'.format((i+1)*multiplier))
        axs[2,i].set_xlabel('time [years]')
        for j in range(2):
            axs[j,i].set(xticks=[])
        for j in range(3):
            axs[j,0].set(ylabel=labels[j])
            axs[j,i].grid('on')
    fig.legend(labels=well_names, loc='right', bbox_to_anchor=(0.95, 0.5))   

def make_dynamic_animation(static, dynamic, ncols=11, multiplier=10, figsize=(20,6), static_label='poro', static_cmap='viridis', interval=100, blit=False):
    labels = [static_label, 'pressure', 'saturation']
    pressure, saturation = dynamic
    tot_frames = pressure.shape[1]
    xlocs, ylocs = [0,0,0,127,127,127], [0,63,127,0,63,127]
    fig, axs = plt.subplots(3, ncols, figsize=figsize, facecolor='white')
    for k in range(ncols):
        r = k*multiplier
        axs[0,k].set(title='#{}'.format(r))
        axs[0,k].imshow(static[r], static_cmap)
        axs[1,k].imshow(pressure[r,0], 'gnuplot2')
        axs[2,k].imshow(saturation[r,0], 'jet')
        for p in range(3):
            axs[p,k].set(xticks=[], yticks=[])
            axs[p,k].scatter(xlocs, ylocs, c='k')
            axs[p,0].set(ylabel=labels[p])
    def animate(i):
        for k in range(ncols):
            r = k*multiplier
            axs[1,k].imshow(pressure[r,i], 'gnuplot2')
            axs[2,k].imshow(saturation[r,i], 'jet')
        return axs[1,k], axs[2,k]
    ani = animation.FuncAnimation(fig, animate, frames=tot_frames, blit=blit, interval=interval)
    ani.save('figures/dynamic_animation.gif')
    plt.show()

def plot_X_observation(data, ncols=10, multiplier=1, figsize=(20,3), cmaps=['gnuplot2','jet']):
    fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=True, sharey=True)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].imshow(data[k,:,:,0].T, cmap=cmaps[0])
        axs[1,i].imshow(data[k,:,:,1].T, cmap=cmaps[1])
        axs[0,i].set(title='Realization {}'.format(k))
    axs[0,i].set_ylabel('Pressure', labelpad=-110, rotation=270)
    axs[1,i].set_ylabel('Saturation', labelpad=-110, rotation=270)
    fig.text(0.5, 0.01, 'Timesteps', ha='center')
    fig.text(0.1, 0.5, 'Location Index', va='center', rotation='vertical')

def plot_X_line_observation(data, times, ncols=10, multiplier=1, figsize=(20,5)):
    fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=True, sharey=True)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].plot(times, data[k,:,:,0])
        axs[1,i].plot(times, data[k,:,:,1])
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(2):
            axs[j,i].set(ylim=[-0.05,1.05])
    axs[0,0].set_ylabel('Pressure')
    axs[1,0].set_ylabel('Saturation')
    fig.text(0.5, 0.04, 'Time [years]', ha='center')

def plot_X_img_observation(data, randx, randy, timing=-1, multiplier=1, ncols=10, figsize=(20,4), cmaps=['gnuplot2', 'jet']):
    fig, axs = plt.subplots(2, ncols, figsize=figsize)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].imshow(data[k,timing,:,:,0], cmap=cmaps[0])
        axs[0,i].scatter(randx, randy, marker='s', c='k')
        axs[1,i].imshow(data[k,timing,:,:,1], cmap=cmaps[1])
        axs[1,i].scatter(randx, randy, marker='s', c='k')
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(2):
            axs[j,i].set(xticks=[], yticks=[])
    axs[0,0].set(ylabel='Pressure')
    axs[1,0].set(ylabel='Saturation')

def plot_data_results(timestamps, true_train, true_test, pred_train, pred_test, channel_select=0, figsize=(20,4), ncols=10, multiplier=1):
    colors = ['k', 'b', 'g']
    fig, axs = plt.subplots(2, ncols, figsize=figsize, facecolor='white')
    for i in range(ncols):
        for k in range(3):
            axs[0,i].plot(timestamps, true_train[i*multiplier,:,k,channel_select], label='P{} true'.format(k+1), linestyle='-', c=colors[k])
            axs[0,i].plot(timestamps, pred_train[i*multiplier,:,k,channel_select], label='P{} pred'.format(k+1), linestyle='--', c=colors[k])
            axs[1,i].plot(timestamps, true_test[i*multiplier,:,k,channel_select], label='P{} true'.format(k+1), linestyle='-', c=colors[k])
            axs[1,i].plot(timestamps, pred_test[i*multiplier,:,k,channel_select], label='P{} pred'.format(k+1), linestyle='--', c=colors[k])
            axs[0,i].set(title='Realization {}'.format(i*multiplier))
    for i in range(1,ncols):
        for j in range(2):
            axs[j,i].set(yticks=[])
    axs[0,0].set(ylabel='Train')
    axs[1,0].set(ylabel='Test')
    fig.text(0.5, 0.04, 'Time [years]', ha='center')
    plt.legend(bbox_to_anchor=(1, 1.5))

def plot_static_results(true, pred, channel_select=0, multiplier=1, cmaps=['jet','gray_r'], figsize=(20,6), ncols=10):
    labs = ['True', 'Pred', 'Difference']
    fig, axs = plt.subplots(3, ncols, figsize=figsize)
    for i in range(ncols):
        k = i*multiplier
        im1 = axs[0,i].imshow(true[k,:,:,channel_select], cmap=cmaps[0], vmin=0, vmax=1); axs[0,0].set(ylabel=labs[0])
        im2 = axs[1,i].imshow(pred[k,:,:,channel_select], cmap=cmaps[0], vmin=0, vmax=1); axs[1,0].set(ylabel=labs[1])
        im3 = axs[2,i].imshow(true[k,:,:,channel_select]-pred[k,:,:,channel_select], cmap=cmaps[1]); axs[2,0].set(ylabel=labs[2])
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(3):
            axs[j,i].set(xticks=[], yticks=[])
    for m in (im1, im2, im3):
        plt.colorbar(m, fraction=0.046, pad=0.04)

def plot_dynamic_results(true, pred, channel_select=0, multiplier=1, ncols=10, figsize=(20,4)):
    labs = ['True', 'Pred', 'Difference']
    if channel_select==0: cmap = 'gnuplot2'
    elif channel_select==1: cmap = 'jet'
    else: cmap = 'binary'
    fig, axs = plt.subplots(3, ncols, figsize=figsize)
    for i in range(10):
        k = i*multiplier
        im1 = axs[0,i].imshow(true[k,:,:,channel_select].T, cmap=cmap)
        im2 = axs[1,i].imshow(pred[k,:,:,channel_select].T, cmap=cmap)
        im3 = axs[2,i].imshow((true[k,:,:,channel_select].T - pred[k,:,:,channel_select].T), cmap='coolwarm')
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(3):
            axs[j,i].set(xticks=[], yticks=[])
            axs[j,0].set(ylabel=labs[j])
    for m in (im1, im2, im3):
        plt.colorbar(m, fraction=0.046, pad=0.04)

def plot_inversion_result(truth, prediction, channel_select=0, multiplier=10, ncols=10, figsize=(20,6), cmaps=['jet','gray_r']):
    labels = ['True','Pred','Difference']
    true, pred = truth[:,:,:,channel_select], prediction[:,:,:,channel_select]
    fig, axs = plt.subplots(3, ncols, figsize=figsize)
    for i in range(ncols):
        k = i*multiplier
        im1 = axs[0,i].imshow(true[k], cmap=cmaps[0], vmin=0, vmax=1)
        im2 = axs[1,i].imshow(pred[k], cmap=cmaps[0], vmin=0, vmax=1)
        im3 = axs[2,i].imshow(true[k]-pred[k], cmap=cmaps[1], vmin=0, vmax=1)
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(3):
            axs[j,0].set(ylabel=labels[j])
            axs[j,i].set(xticks=[], yticks=[])
    plt.colorbar(im1, pad=0.04, fraction=0.046)
    plt.colorbar(im2, pad=0.04, fraction=0.046)
    plt.colorbar(im3, pad=0.04, fraction=0.046)

################################################################################################
######################################### MODEL UTILS ##########################################
################################################################################################
def conv_block(inp, filt, kern=(3,3), pool=(2,2), pad='same'):
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(_)
    _ = InstanceNormalization()(_)
    _ = BatchNormalization()(_)
    _ = GELU()(_)
    _ = AveragePooling2D(pool)(_)
    return _

def decon_block(inp, filt, kern=(3,3), pool=(2,2), pad='same'):
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(_)
    _ = InstanceNormalization()(_)
    _ = BatchNormalization()(_)
    _ = GELU()(_)
    _ = UpSampling2D(pool)(_)
    return _

def make_data_ae(w, code_dim=300, z_dim=10, epochs=100, batch=50, opt=Adam(1e-3)):
    def sample(args, mu=0.0, std=1.0):
        mean, sigma = args
        epsilon = K.random_normal(shape=(K.shape(mean)[0],z_dim), mean=mu, stddev=std)
        return mean + K.exp(sigma)*epsilon
    inputs = Input(shape=(w.shape[1:]))
    shape_b4 = K.int_shape(inputs)[1:]
    _ = Flatten()(inputs)
    _ = Dense(code_dim, activation=PReLU())(_)
    code = _
    _ = Dense(100, activation=PReLU())(_)
    mean = Dense(z_dim)(_)
    sigma = Dense(z_dim)(_)
    latent = Lambda(sample)([mean, sigma])
    z_inp = Input(shape=(z_dim,))
    _ = Dense(100, activation=PReLU())(z_inp)
    _ = Dense(code_dim, activation=PReLU())(_)
    _ = Dense(np.prod(shape_b4), activation='sigmoid')(_)
    out = Reshape(shape_b4)(_)
    enc = Model(inputs, [mean, sigma, latent, code], name='data_encoder')
    dec = Model(z_inp, out, name='data_decoder')
    outputs = dec(enc(inputs)[2])
    vae = Model(inputs, outputs, name='data_vae')
    rec_loss = K.sum(loss_mse(inputs, outputs))*np.prod(w.shape[1:])
    kl_loss  = (-0.5) * K.sum(1 + sigma - K.square(mean) - K.exp(sigma), axis=-1)
    vae_loss = K.mean(rec_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=opt, metrics=['mse'])
    wparams = vae.count_params()
    start = time()
    fit = vae.fit(w, w, epochs=epochs, batch_size=batch, 
                            verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(wparams,traintime))
    return enc, dec, vae, fit

def make_static_ae(y, epochs=300, batch=80, opt=Adam(1e-3), ssim_perc=(2/3)):
    input_static = Input(shape=y.shape[1:])
    _ = conv_block(input_static, 8)
    _ = conv_block(_, 16)
    _ = conv_block(_, 32)
    _ = conv_block(_, 64)
    code = SeparableConv2D(128, (3,3), padding='same', activation='relu')(_)
    shape_b4 = K.int_shape(code)[1:]
    latent = Flatten()(code)
    shape_flat = K.int_shape(latent)[1]
    z_inp = Input(shape=(shape_flat,))
    _ = Reshape(shape_b4)(z_inp)
    _ = decon_block(_, 64)
    _ = decon_block(_, 32)
    _ = decon_block(_, 16)
    _ = decon_block(_, 8)
    output = SeparableConv2D(3, (3,3), padding='same', activation='sigmoid')(_)
    enc = Model(input_static, latent, name='static_encoder')
    dec = Model(z_inp, output, name='static_decoder')
    output_static = dec(enc(input_static))
    ae = Model(input_static, output_static, name='static_ae')
    ssim = 1 - tf.reduce_mean(tf.image.ssim(input_static, output_static, 1.0))
    mse = loss_mse(input_static, output_static)
    dual_loss = (ssim_perc)*ssim + (1-ssim_perc)*mse
    ae.add_loss(dual_loss)
    ae.compile(optimizer=opt, metrics=['mse'])
    yparams = ae.count_params()
    start = time()
    fit = ae.fit(y, y, epochs=epochs, batch_size=batch, verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(yparams,traintime))
    return enc, dec, ae, fit

def make_dynamic_ae(x, code_dim=1000, z_dim=10, epochs=200, batch=80, opt=Adam(1e-3)):
    def sample(args, mu=0.0, std=1.0):
        mean, sigma = args
        epsilon = K.random_normal(shape=(K.shape(mean)[0],z_dim), mean=mu, stddev=std)
        return mean + K.exp(sigma)*epsilon
    inputs = Input(shape=x.shape[1:])
    shape_b4 = K.int_shape(inputs)[1:]
    _ = Flatten()(inputs)
    _ = Dense(code_dim, activation=PReLU())(_)
    code = _
    _ = Dense(100, activation=PReLU())(_)
    mean = Dense(z_dim)(_)
    sigma = Dense(z_dim)(_)
    latent = Lambda(sample)([mean, sigma])
    z_inp = Input(shape=(z_dim,))
    _ = Dense(100, activation=PReLU())(z_inp)
    _ = Dense(code_dim, activation=PReLU())(_)
    _ = Dense(np.prod(shape_b4), activation='sigmoid')(_)
    out = Reshape(shape_b4)(_)
    enc = Model(inputs, [mean,sigma,latent,code], name='dynamic_encoder')
    dec = Model(z_inp, out, name='dynamic_decoder')
    outputs = dec(enc(inputs)[2])
    vae = Model(inputs, outputs, name='dynamic_vae')
    rec_loss = K.sum(loss_mse(inputs, outputs))*np.prod(x.shape[1:])
    kl_loss  = (-0.5) * K.sum(1 + sigma - K.square(mean) - K.exp(sigma), axis=-1)
    vae_loss = K.mean(rec_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=opt, metrics=['mse'])
    xparams = vae.count_params()
    start = time()
    fit = vae.fit(x, x, epochs=epochs, batch_size=batch, 
                        verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(xparams,traintime))
    return enc, dec, vae, fit

def make_ae_prediction(train_true, test_true, ae_model):
    train_pred = ae_model.predict(train_true).astype('float64')
    test_pred  = ae_model.predict(test_true).astype('float64')
    mse_train = img_mse(train_true, train_pred)
    mse_test  = img_mse(train_true, train_pred)
    print('Train MSE: {:.2e} | Test MSE: {:.2e}'.format(mse_train, mse_test))
    if train_true.shape[2]>=7:
        ssim_train = structural_similarity(train_true, train_pred, channel_axis=-1, data_range=1.0)
        ssim_test  = structural_similarity(test_true, test_pred, channel_axis=-1, data_range=1.0)
        print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    else:
        print('Image data must have shape at least (7x7) for ssim calculation')
    return train_pred, test_pred

def make_full_traintest(xtrain, xtest, wtrain, wtest, ytrain, ytest):
    xfull = np.concatenate([xtrain,xtest])
    wfull = np.concatenate([wtrain,wtest])
    yfull = np.concatenate([ytrain,ytest])
    print('X_full: {} | w_full: {} | y_full: {}'.format(xfull.shape, wfull.shape, yfull.shape))
    return xfull, wfull, yfull

def make_inv_regressor(xf, wf, yf, dynamic_enc, data_enc, static_dec, 
                            opt=Adam(1e-5), loss='mse', epochs=500, batch=80):
    dynamic_enc.trainable = False
    data_enc.trainable    = False
    static_dec.trainable  = False
    def dense_block(input, neurons):
        _ = Dense(neurons)(input)
        _ = LayerNormalization()(_)
        _ = BatchNormalization()(_)
        _ = PReLU()(_)
        return _
    x_inp = Input(shape=xf.shape[1:])
    x_latent = dynamic_enc(x_inp)[-1]
    x = dense_block(x_latent, 1000)
    x = dense_block(x, 2000)
    w_inp = Input(shape=wf.shape[1:])
    w_latent = data_enc(w_inp)[-1]
    w = dense_block(w_latent, 300)
    w = dense_block(w, 600)
    w = dense_block(w, 1000)
    _ = Concatenate()([x, w])
    _ = LayerNormalization()(_)
    _ = dense_block(_, 5000)
    _ = dense_block(_, 8*8*128)
    out = static_dec(_)
    reg = Model([x_inp, w_inp], out)
    rparams = reg.count_params()
    reg.compile(optimizer=opt, loss=loss, metrics=['mse'])
    start = time()
    fit = reg.fit([xf, wf], yf, epochs=epochs, batch_size=batch, verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(rparams,traintime))
    return reg, fit

def make_inv_prediction(regmodel, x_tuple, w_tuple, y_tuple):
    xtrain, xtest = x_tuple
    wtrain, wtest = w_tuple
    ytrain, ytest = y_tuple
    inv_train = regmodel.predict([xtrain, wtrain]).astype('float64')
    inv_test = regmodel.predict([xtest, wtest]).astype('float64')
    mse_train, mse_test = img_mse(ytrain, inv_train), img_mse(ytest, inv_test)
    print('Train MSE: {:.2e} | Test MSE: {:.2e}'.format(mse_train, mse_test))
    ssim_train = structural_similarity(ytrain, inv_train, channel_axis=-1, data_range=1.0)
    ssim_test  = structural_similarity(ytest, inv_test, channel_axis=-1, data_range=1.0)
    print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    return inv_train, inv_test

def make_fwd_regressor(xf, wf, yf, dynamic_dec, data_dec, static_enc, 
                       drop=0.2, opt=Adam(1e-3), loss='mse', epochs=500, batch=70):
    dynamic_dec.trainable = False
    data_dec.trainable = False
    static_enc.trainable = False
    def dense_block(input, neurons):
        _ = Dense(neurons, activity_regularizer='l2')(input)
        _ = LayerNormalization()(_)
        _ = BatchNormalization()(_)
        _ = Dropout(drop)(_)
        _ = PReLU()(_)
        return _
    y_inp = Input(shape=(yf.shape[1:]))
    y_latent = static_enc(y_inp)
    y_x = dense_block(y_latent, 100)
    y_x = dense_block(y_x, 10)
    x_out = dynamic_dec(y_x)
    y_w = dense_block(y_latent, 100)
    y_w = dense_block(y_w, 10)
    w_out = data_dec(y_w)
    fwd = Model(y_inp, [x_out, w_out])
    rparams = fwd.count_params()
    fwd.compile(optimizer=opt, loss=loss, metrics=['mse'])
    start = time()
    fit = fwd.fit(yf, [xf, wf], epochs=epochs, batch_size=batch, verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(rparams, traintime))
    return fwd, fit

def make_fwd_prediction(fwdmodel, x_tuple, w_tuple, y_tuple):
    xtrain, xtest = x_tuple
    wtrain, wtest = w_tuple
    ytrain, ytest = y_tuple
    fwd_train = fwdmodel.predict(ytrain)
    fwd_test = fwdmodel.predict(ytest)
    fwd_x_train, fwd_w_train = fwd_train
    fwd_x_test,  fwd_w_test  = fwd_test
    mse_x_train = img_mse(xtrain, fwd_x_train)
    mse_x_test  = img_mse(xtest,  fwd_x_test)
    mse_w_train = img_mse(wtrain, fwd_w_train)
    mse_w_test  = img_mse(wtest,  fwd_w_test)
    print('X - MSE: Train {:.2e} | Test: {:.2e}'.format(mse_x_train, mse_x_test))
    print('w - MSE: Train {:.2e} | Test: {:.2e}'.format(mse_w_train, mse_w_test))
    fwd_X = {'train': fwd_x_train, 'test': fwd_x_test}
    fwd_W = {'train': fwd_w_train, 'test': fwd_w_test}
    return fwd_X, fwd_W
