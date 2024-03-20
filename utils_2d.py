################################################################################################
########################################## INITIALIZE ##########################################
################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from time import time
from scipy.io import loadmat
from scipy.interpolate import griddata

from sklearn.metrics import r2_score, mean_squared_error
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity

import keras.backend as K
from keras import Model, Input
from tensorflow_addons.layers import InstanceNormalization, GELU
from keras.layers import BatchNormalization, LayerNormalization, PReLU
from keras.layers import Flatten, Reshape, Concatenate, Lambda
from keras.layers import SeparableConv2D, AveragePooling2D, UpSampling2D, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss_mse
from keras.losses import mean_absolute_error as loss_mae

################################################################################################
########################################## DATALOADER ##########################################
################################################################################################
n_realizations = 1000
n_timesteps    = 45
xy_dim         = 128
showfig        = True
savefig        = True
savefolder     = 'paper/figures'

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

def my_train_test_split(X, y, w, nobs, split_perc=0.7, n_realizations=n_realizations, xy_dim=xy_dim, equigrid:bool=False):
    train_size = int(np.ceil(n_realizations*split_perc))
    def perfect_square(x):
        if x>= 0:
            sr = int(np.sqrt(x))
            return (sr*sr)==x
        return False
    if equigrid:
        assert perfect_square(nobs), 'Number of observations must be a perfect square for equigrid splitting'
        sqrt_obs = int(np.sqrt(nobs))
        randl = np.linspace(0, xy_dim-1, sqrt_obs, dtype=int)
        randx, randy = [r.flatten() for r in np.meshgrid(randl, randl)]
    else:
        randx, randy = [np.random.randint(xy_dim, size=nobs) for _ in range(2)]
    X_train, X_test = X[:train_size,:,randx,randy], X[train_size:,:,randx,randy]
    w_train, w_test = w[:train_size], w[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print('X_train shape: {}   | X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('w_train shape: {}    | w_test shape: {}'.format(w_train.shape, w_test.shape))
    print('y_train shape: {} | y_test shape: {}'.format(y_train.shape, y_test.shape))
    return [X_train, X_test], [y_train, y_test], [w_train, w_test], [randx, randy]

def make_inv_backnorm(inv_tuple, true_tuple):
    inv_train, inv_test = inv_tuple
    inv_all = np.concatenate([inv_train, inv_test])
    poro, perm, channels = true_tuple
    poro_hat = my_normalize(inv_all[...,0], poro, mode='inverse')
    perm_hat = my_normalize(inv_all[...,1], perm, mode='inverse')
    chan_hat = my_normalize(inv_all[...,2], channels, mode='inverse')
    return poro_hat, perm_hat, chan_hat

def make_fwd_X_backnorm(fwd_tuple, true_tuple):
    fwd_train, fwd_test = fwd_tuple
    fwd_all = np.concatenate([fwd_train, fwd_test])
    pressure, saturation = true_tuple
    pres_hat = my_normalize(fwd_all[...,0], pressure, mode='inverse')
    satu_hat = my_normalize(fwd_all[...,1], saturation, mode='inverse')
    return pres_hat, satu_hat

def make_fwd_w_backnorm(fwd_tuple, true_tuple):
    fwd_train, fwd_test = fwd_tuple
    fwd_all = np.concatenate([fwd_train, fwd_test])
    well_opr, well_wpr, well_wcut = true_tuple
    opr_hat = my_normalize(fwd_all[...,0], well_opr, mode='inverse')
    wpr_hat = my_normalize(fwd_all[...,1], well_wpr, mode='inverse')
    wcut_hat = my_normalize(fwd_all[...,2], well_wcut, mode='inverse')
    return opr_hat, wpr_hat, wcut_hat

def fwd_latent_spatial_interpolation(fwdX, locs, n_obs:int=100, train_or_test:str='train', method:str='cubic'):
    randx, randy = locs
    X, Y, = np.meshgrid(np.arange(128), np.arange(128))
    s = np.zeros((n_obs,45,128,128,2))
    for i in range(n_obs):
        for t in range(45):
            for c in range(2):
                s[i,t,:,:,c] = griddata((randy,randx), fwdX[train_or_test][i,t,:,c], (X,Y), method=method)
    return s

################################################################################################
########################################## PLOT UTILS ##########################################
################################################################################################
def plot_loss(fit, title='', figsize=None, savefig:bool=False, fname:str=None):
    if figsize:
        plt.figure(figsize=figsize)
    loss, val = fit.history['loss'], fit.history['val_loss']
    epochs, iterations = len(loss), np.arange(len(loss))
    plt.plot(iterations, loss, '-', label='loss')
    plt.plot(iterations, val, '-', label='validation loss')
    plt.title(title+' Training: Loss vs epochs')
    plt.legend(facecolor='lightgrey', edgecolor='k'); plt.grid(True, which='both')
    plt.ylabel('Epochs'); plt.ylabel('Loss')
    plt.xticks(iterations[::epochs//10])
    if savefig:
        assert fname is not None, 'Please provide a filename for the figure'
        plt.savefig('{}/{}.png'.format(savefolder, fname))
    return None

def plot_loss_all(loss1, loss2, loss3, title1:str='Data', title2:str='Static', title3:str='Dynamic', figsize=(10,4)):
    plt.figure(figsize=figsize, facecolor='white')
    plt.subplot(131); plot_loss(loss1, title=title1)
    plt.subplot(132); plot_loss(loss2, title=title2)
    plt.subplot(133); plot_loss(loss3, title=title3)
    plt.tight_layout()
    plt.savefig('{}/trainingperformance_all.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_static(poro, perm, channels, multiplier:int=1, ncols:int=10, inv_flag:bool=False,
                figsize=(20,4), cmaps=['binary', 'viridis', 'jet']):
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
    plt.tight_layout()
    if inv_flag:
        plt.savefig('{}/inverted_geomodels-plot_static.png'.format(savefolder)) if savefig else None
    else:
        plt.savefig('{}/simulated_geomodels-plot_static.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_dynamic(static, dynamic, multiplier:int=1, nrows:int=5, figsize=(30,8), cmaps=['jet', 'gnuplot2']):
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
    plt.tight_layout()
    plt.savefig('{}/simulated_dynamic-plot_dynamic.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_data(timestamps, opr, wpr, wcut, multiplier:int=1, ncols:int=10, figsize=(20,6)):
    labels     = ['Oil Rate [bpd]', 'Water Rate [bpd]', 'Water Cut [%]']
    well_names = ['P1', 'P2', 'P3']
    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=figsize, facecolor='white')
    for i in range(ncols):
        axs[0,i].plot(timestamps, opr[i*multiplier])
        axs[1,i].plot(timestamps, wpr[i*multiplier])
        axs[2,i].plot(timestamps, wcut[i*multiplier])
        axs[0,i].set_title('realization {}'.format((i+1)*multiplier))
        axs[2,i].set_xlabel('time [years]')
        axs[2,i].set_ylim(0,1)
        for j in range(2):
            axs[j,i].set(xticks=[])
        for j in range(3):
            axs[j,0].set(ylabel=labels[j])
            axs[j,i].grid('on')
    fig.legend(labels=well_names, loc='right', bbox_to_anchor=(0.95, 0.5))   
    plt.tight_layout()
    plt.savefig('{}/simulated_wells-plot_data.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def make_dynamic_animation(static, dynamic, ncols:int=11, multiplier:int=10, figsize=(20,6), 
                           static_label:str='poro', static_cmap:str='viridis', interval=100, blit:bool=False):
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
    plt.show() if showfig else None
    return None

def plot_X_img_observation(data, randx, randy, timing=-1, multiplier:int=1, ncols:int=10, figsize=(20,4), cmaps=['gnuplot2', 'jet']):
    fig, axs = plt.subplots(2, ncols, figsize=figsize)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].imshow(data[k,timing,:,:,0], cmap=cmaps[0])
        axs[0,i].scatter(randx, randy, marker='s', c='k')
        axs[1,i].imshow(data[k,timing,:,:,1], cmap=cmaps[1])
        axs[1,i].scatter(randx, randy, marker='s', c='k')
        axs[0,i].set_title('Realization {}'.format(k), weight='bold')
        for j in range(2):
            axs[j,i].set(xticks=[], yticks=[])
    axs[0,0].set_ylabel('Pressure', weight='bold')
    axs[1,0].set_ylabel('Saturation', weight='bold')
    plt.tight_layout()
    plt.savefig('{}/X_img_obersevations.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_X_observation(data, ncols:int=10, multiplier:int=1, figsize=(20,3), cmaps=['gnuplot2','jet']):
    fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=True, sharey=True)
    for i in range(ncols): 
        k = i*multiplier
        axs[0,i].imshow(data[k,:,:,0].T, cmap=cmaps[0])
        axs[1,i].imshow(data[k,:,:,1].T, cmap=cmaps[1])
        axs[0,i].set_title('Realization {}'.format(k), weight='bold')
    axs[0,i].set_ylabel('Pressure', labelpad=-133, rotation=270, weight='bold')
    axs[1,i].set_ylabel('Saturation', labelpad=-133, rotation=270, weight='bold')
    fig.text(0.5, 0, 'Timesteps', ha='center', weight='bold')
    fig.text(0, 0.5, 'Location Index', va='center', rotation='vertical', weight='bold')
    plt.tight_layout()
    plt.savefig('{}/X_obersevations.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_X_line_observation(data, times, ncols:int=10, multiplier:int=1, figsize=(20,5)):
    fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=True, sharey=True)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].plot(times, data[k,:,:,0])
        axs[1,i].plot(times, data[k,:,:,1])
        axs[0,i].set_title('Realization {}'.format(k), weight='bold')
        for j in range(2):
            axs[j,i].set(ylim=[-0.05,1.05])
    axs[0,0].set_ylabel('Pressure', weight='bold')
    axs[1,0].set_ylabel('Saturation', weight='bold')
    fig.text(0.5, 0, 'Time [years]', ha='center', weight='bold')
    plt.tight_layout()
    plt.savefig('{}/X_line_obersevations.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_data_results(timestamps, true_train, true_test, pred_train, pred_test, channel_select:int=0, 
                      ncols:int=10, multiplier:int=1, figsize=(20,4)):
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
    plt.tight_layout()
    plt.savefig('{}/well_predictions-plot_data_results.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_static_results(true, pred, train_or_test:str='train', 
                        channel_select:int=0, multiplier:int=1, ncols:int=10, cmaps=['viridis','gray_r'], figsize=(20,6)):
    labs = ['True', 'Pred', 'Difference']
    fig, axs = plt.subplots(3, ncols, figsize=figsize)
    if train_or_test == 'train':
        true, pred = true[0], pred[0]
    elif train_or_test=='test':
        true, pred = true[1], pred[1]
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
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
    plt.tight_layout()
    plt.savefig('{}/model_predictions_{}-plot_static_results.png'.format(savefolder, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_dynamic_results(true, pred, train_or_test:str='train', channel_select:int=1, multiplier:int=1, ncols:int=10, figsize=(20,4)):
    labs = ['True', 'Pred', 'Difference']
    if train_or_test=='train':
        true, pred = true[0], pred[0]
    elif train_or_test=='test':
        true, pred = true[1], pred[1]
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
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
    plt.tight_layout()
    plt.savefig('{}/dynamic_predictions_{}-plot_dynamic_results.png'.format(savefolder, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_inversion_result(truth, prediction, train_or_test:str='train',
                          channel_select:int=0, multiplier:int=10, ncols:int=10, figsize=(20,6), cmaps=['viridis','gray_r']):
    if train_or_test=='train':
        truth, prediction = truth[0], prediction[0]
    elif train_or_test=='test':
        truth, prediction = truth[1], prediction[1]
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
    if channel_select==0:
        dname = 'Porosity'
    elif channel_select==1:
        dname = 'Permeability'
    elif channel_select==2:
        dname = 'Facies'
    else:
        raise ValueError('channel_select must be either 0, 1 or 2')    
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
    plt.tight_layout()
    plt.savefig('{}/inv_model_prediction_{}_{}'.format(savefolder, dname, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_results_data(wtrue, wfwd, train_or_test:str='train', channel_select:int=2, multiplier:int=1, 
                          ncols:int=10, nrows:int=3, colors=['tab:blue','tab:orange','tab:green'], figsize=(15,7.5)):
    if train_or_test=='train':
        wtrue, wfwd = wtrue[0], wfwd['train']
    elif train_or_test=='test':
        wtrue, wfwd = wtrue[1], wfwd['test']
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
    plt.figure(figsize=figsize)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            plt.subplot(nrows, ncols, k+1)
            plt.grid(True, which='both')
            plt.title('R{}'.format(k*multiplier))
            for t in range(3):
                plt.plot(wtrue[k*multiplier,:,t,channel_select], color=colors[t], label='W#{}'.format(t))
                plt.plot(wfwd[k*multiplier,:,t,channel_select], color=colors[t], linestyle='--')
            k += 1
    plt.tight_layout()
    plt.savefig('{}/fwd_well_prediction_{}'.format(savefolder, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_results_dynamic(xtrue, xfwd, train_or_test:str='train', multiplier:int=1, ncols:int=6, cmaps=['gnuplot2','jet'], figsize=(15,8)):
    if train_or_test=='train':
        xtrue, xfwd = xtrue[0], xfwd['train']
    elif train_or_test=='test':
        xtrue, xfwd = xtrue[1], xfwd['test']
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, ncols, figure=fig)
    ax1, ax2, ax3, ax4 = [], [], [], []
    for j in range(ncols):
        ax1.append(fig.add_subplot(gs[0, j]))
        ax2.append(fig.add_subplot(gs[1, j]))
        ax3.append(fig.add_subplot(gs[2, j]))
        ax4.append(fig.add_subplot(gs[3, j]))
        ax1[j].imshow(xtrue[j*multiplier,:,:,0].T, cmap=cmaps[0])
        ax2[j].imshow(xfwd[j*multiplier,:,:,0].T, cmap=cmaps[0])
        ax3[j].imshow(xtrue[j*multiplier,:,:,1].T, cmap=cmaps[1])
        ax4[j].imshow(xfwd[j*multiplier,:,:,1].T, cmap=cmaps[1])
        for ax in [ax1,ax3]:
            ax[j].set_title('R{}'.format(j*multiplier))
    ax1[0].set_ylabel('True', weight='bold')
    ax2[0].set_ylabel('Pred', weight='bold', color='red')
    ax3[0].set_ylabel('True', weight='bold')
    ax4[0].set_ylabel('Pred', weight='bold', color='red')
    plt.tight_layout()
    plt.savefig('{}/fwd_dynamic_prediction_{}'.format(savefolder, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_results_dynamic_line(xtrue, xfwd, train_or_test:str='train', nrows:int=3, multiplier:int=1, nw:int=5, figsize=(15,8), colors=None):
    if train_or_test == 'train':
        xtrue, xfwd = xtrue[0], xfwd['train']
    elif train_or_test == 'test':
        xtrue, xfwd = xtrue[1], xfwd['test']
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
    if colors is None:
        colors = ['tab:blue','tab:orange','tab:green', 'tab:red', 'tab:purple', 
                  'tab:pink', 'tab:olive', 'tab:cyan', 'tab:brown', 'black']
    labels = ['Pressure','Saturation']
    idx = np.sort(np.random.choice(range(xtrue.shape[2]), size=nw, replace=False))
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(nrows, 2, figure=fig)
    axs = {}
    for i in range(nrows):
        for j in range(2):
            axs[i,j] = fig.add_subplot(gs[i,j])
            for k in range(nw):
                axs[i,j].plot(xtrue[i*multiplier,:,idx[k],j], color=colors[k], label='W#{}'.format(idx[k]))
                axs[i,j].plot(xfwd[i*multiplier,:,idx[k],j], color=colors[k], linestyle='--')
            axs[i,j].set_title('R{}'.format(i*multiplier))
            axs[i,j].set_ylabel(labels[j], weight='bold')
            axs[i,j].grid(True, which='both')
    for j in range(2):
        axs[nrows-1,j].set_xlabel('Time')
        axs[nrows-1,j].legend(loc='lower center', facecolor='lightgrey', edgecolor='k',
                              ncols=nw, bbox_to_anchor=(0.5,-0.5))
    plt.tight_layout()
    plt.savefig('{}/fwd_dynamic_line_prediction_{}'.format(savefolder, train_or_test)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_inv_latent_dashboard(xtrue, wtrue, ypred, realization:int=0, 
                              figsize=(10,6), colors=None, xcmaps=None, ycmaps=None):
    if colors is None:
        colors = ['tab:blue','tab:orange','tab:green']
    if xcmaps is None:
        xcmaps = ['gnuplot2','jet']
    if ycmaps is None:
        ycmaps = ['viridis','jet','binary']
    xlabs = ['Pressure', 'Saturation']
    wnames = ['P1','P2','P3']
    wlabs = ['OPR','WPR','WCUT']
    ylabs = ['Porosity','Permeability','Facies']
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(6,3,figure=fig)
    ax11 = fig.add_subplot(gs[0:2,0])
    ax12 = fig.add_subplot(gs[2:4,0])
    ax13 = fig.add_subplot(gs[4:6,0])
    ax1  = [ax11, ax12, ax13]
    ax21 = fig.add_subplot(gs[0:3,1])
    ax22 = fig.add_subplot(gs[3:6,1])
    ax2  = [ax21, ax22]
    ax31 = fig.add_subplot(gs[0:2,2])
    ax32 = fig.add_subplot(gs[2:4,2])
    ax33 = fig.add_subplot(gs[4:6,2])
    ax3  = [ax31, ax32, ax33]
    for i in range(3):
        for j in range(3):
            ax1[j].plot(wtrue[realization,:,i,j], color=colors[i], label=wnames[i])
        ax1[j].legend(loc='lower right')
        ax1[i].grid(True, which='both')
        ax1[i].set_ylabel(wlabs[i], weight='bold')
    ax1[0].set_title('r(t)', weight='bold')
    ax13.set_xlabel('Time'); ax11.set_xticklabels([]); ax12.set_xticklabels([])
    for i in range(2):
        im = ax2[i].imshow(xtrue[realization,:,:,i].T, cmap=xcmaps[i], aspect='auto', vmin=0, vmax=1)
        ax2[i].set(ylabel='Location index')
        cb = plt.colorbar(im, ax=ax2[i], pad=0.04, fraction=0.046)
        cb.set_label(xlabs[i], labelpad=15, rotation=270, weight='bold')
    ax2[0].set_title('d(t)', weight='bold')
    ax21.set_xticklabels([]); ax22.set_xlabel('Time')
    for i in range(3):
        im = ax3[i].imshow(ypred[realization,:,:,i], cmap=ycmaps[i], aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax3[i], pad=0.04, fraction=0.046)
        ax3[i].set(xticks=[], yticks=[])
        ax3[i].set_xlabel(ylabs[i], weight='bold')
    ax3[0].set_title('$\hat{m}=g(d,r)$', weight='bold')
    plt.tight_layout()
    plt.savefig('{}/dashboard_inverse_{}.png'.format(savefolder, realization)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_latent_dashboard(wtrue, ytrue, fwdX, fwdw, 
                          figsize=(10,6), realization:int=0, colors=None, xcmaps=None, ycmaps=None):
    if colors is None:
        colors = ['tab:blue','tab:orange','tab:green']
    if xcmaps is None:
        xcmaps = ['gnuplot2','jet']
    if ycmaps is None:
        ycmaps = ['viridis','jet','binary']
    xlabs = ['Pressure', 'Saturation']
    wnames = ['P1','P2','P3']
    wlabs = ['OPR','WPR','WCUT']
    ylabs = ['Porosity','Permeability','Facies']
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(6,3, figure=fig)
    ax11 = fig.add_subplot(gs[0:2,0])
    ax12 = fig.add_subplot(gs[2:4,0])
    ax13 = fig.add_subplot(gs[4:6,0])
    ax1  = [ax11, ax12, ax13]
    ax21 = fig.add_subplot(gs[0:2,1])
    ax22 = fig.add_subplot(gs[2:4,1])
    ax23 = fig.add_subplot(gs[4:6,1])
    ax2  = [ax21, ax22, ax23]
    ax31 = fig.add_subplot(gs[0:3,2])
    ax32 = fig.add_subplot(gs[3:6,2])
    ax3  = [ax31, ax32]
    for i in range(3):
        im = ax1[i].imshow(ytrue[realization,:,:,i], cmap=ycmaps[i], aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1[i], pad=0.04, fraction=0.046)
        ax1[i].set(xticks=[], yticks=[]); ax1[i].set_xlabel(ylabs[i], weight='bold')
    ax11.set_title('$m$', weight='bold')
    for i in range(3):
        for j in range(3):
            ax2[j].plot(wtrue[realization,:,i,j], color=colors[i], label=wnames[j])
            ax2[j].plot(fwdw[realization,:,i,j], color=colors[i], linestyle='--')
        ax2[j].legend(loc='lower right')
        ax2[i].grid(True, which='both')
        ax2[i].set_ylabel(wlabs[i], weight='bold')
    ax21.set_title('$r(t)=\hat{f}_1(m)$', weight='bold')
    ax23.set_xlabel('time'); ax21.set_xticklabels([]); ax22.set_xticklabels([])
    for i in range(2):
        im = ax3[i].imshow(fwdX[realization,:,:,i].T, cmap=xcmaps[i], aspect='auto', vmin=0, vmax=1)
        cb = plt.colorbar(im, ax=ax3[i], pad=0.04, fraction=0.046)
        cb.set_label(xlabs[i], weight='bold', rotation=270, labelpad=15)
        ax3[i].set(ylabel='Location index')
    ax31.set_title('$d(t)=\hat{f}_2(m)$', weight='bold')
    ax31.set_xticklabels([]); ax32.set_xlabel('time')
    for ax in ax2:
        ax.set(xlim=(0,45))
    plt.tight_layout()
    plt.savefig('{}/dashboard_forward_{}.png'.format(savefolder, realization)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_latent_map(s, ytrue, xtrue, locs, realization:int=1, xchannel:int=1, ychannel:int=0,
                        ms=15, m='s', minj='v', mprod='^', vmin=0.2, vmax=0.8, figsize=(12,10)):
    randx, randy = locs
    injx,  injy  = np.array([0,   0,   0]),   np.array([0, 63, 127])
    prodx, prody = np.array([127, 127, 127]), np.array([0, 63, 127])
    ylabels = ['Porosity','Permeability','Facies']
    ycmaps  = ['viridis','jet','binary']
    xlabels = ['Pressure','Saturation']
    xcmaps  = ['gnuplot2','jet']
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(2, 1, height_ratios=[1,2.33])
    gs1 = GridSpec(1, 5, figure=subfigs[0])
    gs2 = GridSpec(3, 5, figure=subfigs[1], width_ratios=[1,1,1,1,0.1])
    subfigs[0].subplots_adjust(wspace=0.5)
    cax = subfigs[1].add_subplot(gs2[:, -1])
    for j in range(3):
        ax = subfigs[0].add_subplot(gs1[0, j])
        ax.imshow(ytrue[realization,...,j], cmap=ycmaps[j])
        ax.set_title(ylabels[j], weight='bold')
        ax.set(xticklabels=[], yticklabels=[])
        ax.set_ylabel('Ground Truth', weight='bold', fontsize=12) if j==0 else None
        ax.scatter(injx, injy, s=150, c='b', marker=minj, edgecolor='k')
        ax.scatter(prodx, prody, s=150, c='r', marker=mprod, edgecolor='k')
    for p, j in enumerate(range(3,5)):
        ax = subfigs[0].add_subplot(gs1[0, j])
        ax.imshow(xtrue[realization,-1,:,:,p], cmap=xcmaps[p])
        ax.set_title(xlabels[p]+' @ t=45', weight='bold')
        ax.set(xticklabels=[], yticklabels=[])
        ax.scatter(injx, injy, s=150, c='b', marker=minj, edgecolor='k')
        ax.scatter(prodx, prody, s=150, c='r', marker=mprod, edgecolor='k')
    k = 0
    for i in range(3):
        for j in range(4):
            ax = subfigs[1].add_subplot(gs2[i, j])
            ax.scatter(randy, randx, s=ms, marker=m, c='k')
            ax.scatter(injx, injy, s=150, c='b', marker=minj, edgecolor='k')
            ax.scatter(prodx, prody, s=150, c='r', marker=mprod, edgecolor='k')
            im = ax.imshow(s[realization,k,:,:,xchannel], cmap=xcmaps[xchannel], vmin=vmin, vmax=vmax, aspect='auto')
            ax.imshow(ytrue[realization,...,ychannel], cmap=ycmaps[ychannel], alpha=0.3)
            ax.set(xticklabels=[], yticklabels=[])
            ax.set_title('t={}'.format(k+1), weight='bold')
            k += 4
    subfigs[1].text(0.03, 0.5, 'Predicted Saturation distribution over time', 
                    ha='center', va='center', weight='bold', fontsize=14, rotation=90)
    subfigs[0].text(0.5, 1.025, 'Realization {}'.format(realization), 
                    weight='bold', fontsize=16, ha='center', va='center', 
                    bbox=dict(edgecolor='k', facecolor='w'))
    plt.colorbar(im, cax=cax, pad=0.04, fraction=0.046)
    plt.tight_layout()
    plt.savefig('{}/fwd_spatial_predictions_{}.png'.format(savefolder, realization)) if savefig else None
    plt.show() if showfig else None
    return None

def plot_fwd_resoil(true, pred, percentiles=[10,50,90], correction=0.75, bins:int=25, n_obs=100, 
                    cmap='inferno', s=50, m='s', figsize=(12,4), return_data:bool=True):
    xtrue = np.sum(true[:n_obs,...,-1], axis=(1,2,3))/(xy_dim*xy_dim)
    xpred = np.sum(np.nan_to_num(pred[...,-1], nan=correction), axis=(1,2,3))/(xy_dim*xy_dim)
    print('R2: {:.4f} | MSE: {:.4f}'.format(r2_score(xtrue, xpred), mean_squared_error(xtrue, xpred))) 
    xcum_true_uq = np.percentile(xtrue, percentiles)
    xcum_pred_uq = np.percentile(xpred, percentiles)
    uq = {'True': xcum_true_uq, 'Predicted': xcum_pred_uq}
    print('P  - True: {} | Predicted: {}'.format(np.round(xcum_true_uq,3), np.round(xcum_pred_uq,3)))
    print('UQ - True: {} | Predicted: {}'.format(np.round(xcum_true_uq[2]-xcum_true_uq[0],4),
                                                 np.round(xcum_pred_uq[2]-xcum_pred_uq[0],4)))
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.hist(xtrue, bins=bins, label='True', density=True, edgecolor='k')
    plt.hist(xpred, bins=bins, label='Predicted', density=True, edgecolor='k', alpha=0.6)
    plt.xlabel('Average Remaining Oil Saturation [%]', weight='bold')
    plt.ylabel('Density', weight='bold')
    plt.grid(True, which='both', alpha=0.5)
    plt.legend(facecolor='lightgrey', edgecolor='k')
    plt.subplot(122)
    im = plt.scatter(xtrue, xpred, c=range(n_obs), s=s, marker=m, edgecolor='k', alpha=0.85, cmap=cmap, vmin=0, vmax=100)
    cb = plt.colorbar(im, pad=0.04, fraction=0.046)
    cb.set_label('Realization #', labelpad=15, rotation=270, weight='bold')
    start, end = np.floor(np.min([xtrue,xpred])), np.ceil(np.max([xtrue,xpred]))
    plt.xlim(start, end); plt.ylim(start, end)
    plt.axline([start, start], [end, end], color='k', linestyle='--')
    plt.grid(True, which='both', alpha=0.5)
    plt.xlabel('True', weight='bold'); plt.ylabel('Predicted', weight='bold')
    plt.title('Average Remaining Oil Saturation [%]', weight='bold')
    plt.tight_layout()
    plt.savefig('{}/fwd_resoil_uncertainty.png'.format(savefolder)) if savefig else None
    plt.show() if showfig else None
    if return_data:
        return xtrue, xpred, uq

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

def make_data_ae(w, code_dim:int=300, z_dim:int=10, epochs:int=100, batch:int=50, opt=Adam(1e-3)):
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

def make_static_ae(y, epochs:int=300, batch:int=80, opt=Adam(1e-3), ssim_perc=(2/3)):
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

def make_dynamic_ae(x, code_dim:int=1000, z_dim:int=10, epochs:int=200, batch:int=80, opt=Adam(1e-3)):
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
                            opt=Adam(1e-5), loss='mse', epochs:int=500, batch:int=80):
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

def make_fwd_regressor(xf, wf, yf, dynamic_dec, data_dec, static_enc, latent_dim:int=10,
                       opt=Adam(1e-3), epochs:int=100, batch:int=80):
    dynamic_dec.trainable = False
    data_dec.trainable    = False
    static_enc.trainable  = False
    def dense_block(input, neurons):
        _ = Dense(neurons)(input)
        _ = BatchNormalization()(_)
        _ = GELU()(_)
        return _
    def sample(args, mu=0.0, std=1.0):
        mean, sigma = args
        epsilon = K.random_normal(shape=(K.shape(mean)[0], latent_dim), mean=mu, stddev=std)
        return mean + K.exp(sigma) * epsilon
    y_inp = Input(shape=(yf.shape[1:]))
    y_latent = static_enc(y_inp)
    y_latent = dense_block(y_latent, 2048)
    y_latent = dense_block(y_latent, 512)
    y_latent = dense_block(y_latent, 128)
    # dynamic field data (x)
    x_inp = Input(shape=(xf.shape[1:]))
    x_mu = Dense(latent_dim)(y_latent)
    x_sd = Dense(latent_dim)(y_latent)
    x_z = Lambda(sample)([x_mu, x_sd])
    x_out = dynamic_dec(x_z)
    x_mse = K.sum(loss_mse(x_inp, x_out))*np.prod(xf.shape[1:])
    x_kl_loss = -0.5 * K.sum(1 + x_sd - K.square(x_mu) - K.exp(x_sd), axis=-1)
    x_loss = K.mean(x_mse + x_kl_loss)
    # dynamic well data (w)
    w_inp = Input(shape=(wf.shape[1:]))
    w_mu = Dense(latent_dim)(y_latent)
    w_sd = Dense(latent_dim)(y_latent)
    w_z = Lambda(sample)([w_mu, w_sd])
    w_out = data_dec(w_z)
    w_mse = K.sum(loss_mse(w_inp, w_out))
    w_mae = K.sum(loss_mae(w_inp, w_out))
    w_ln_loss = K.mean(w_mse + w_mae)*np.prod(wf.shape[1:])
    w_kl_loss = -0.5 * K.sum(1 + w_sd - K.square(w_mu) - K.exp(w_sd), axis=-1)
    w_loss = K.mean(w_ln_loss + w_kl_loss)
    # inverse forward model
    fwd = Model([x_inp, w_inp, y_inp], [x_out, w_out])
    fwd.add_loss(K.mean(x_loss + w_loss))
    # compile and train
    rparams = fwd.count_params()
    fwd.compile(optimizer=opt, metrics=['mse'])
    start = time()
    fit = fwd.fit([xf,wf,yf], [xf, wf], epochs=epochs, batch_size=batch, verbose=0, validation_split=0.2)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(rparams, traintime))
    return fwd, fit

def make_fwd_prediction(fwdmodel, x_tuple, w_tuple, y_tuple):
    xtrain, xtest = x_tuple
    wtrain, wtest = w_tuple
    ytrain, ytest = y_tuple
    fwd_train = fwdmodel.predict([xtrain,wtrain,ytrain])
    fwd_test = fwdmodel.predict([xtest,wtest,ytest])
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

def save_models(encoders, decoders, autoencoders, regressors, folder:str='models_2D'):
    # encoders
    static_enc, dynamic_enc, data_enc = encoders
    static_enc.save('{}/static_enc.keras'.format(folder))
    dynamic_enc.save('{}/dynamic_enc.keras'.format(folder))
    data_enc.save('{}/data_enc.keras'.format(folder))
    # decoders
    static_dec, dynamic_dec, data_dec = decoders
    static_dec.save('{}/static_dec.keras'.format(folder))
    dynamic_dec.save('{}/dynamic_dec.keras'.format(folder))
    data_dec.save('{}/data_dec.keras'.format(folder))
    # autoencoders
    static_ae, dynamic_ae, data_ae = autoencoders
    static_ae.save('{}/static_ae.keras'.format(folder))
    dynamic_ae.save('{}/dynamic_ae.keras'.format(folder))
    data_ae.save('{}/data_ae.keras'.format(folder))
    # latent regressors
    inv_reg, fwd_reg = regressors
    inv_reg.save('{}/inv_latent.keras'.format(folder))
    fwd_reg.save('{}/fwd_latent.keras'.format(folder))
    # done
    print('All models saved to {}'.format(folder))
    return None

################################################################################################
######################################## END OF SCRIPT #########################################
################################################################################################

if __name__ == '__main__':
    showfig = False
    sample_realization = 14
    time0 = time()

    print('--------------------------------------------------------------------------')
    print('Latent space variational geologic inversion from multi-source dynamic data')
    print('--------------------------------------------------------------------------')
    print('Module: utils_2d.py | Direct Execution')
    print('-------------------------------------')

    ### Initialize script
    K.clear_session()
    check_tensorflow_gpu()
    
    ### Load and visualize data
    timestamps, poro, perm, channels, pressure, saturation, well_opr, well_wpr, well_wcut = load_initial_data()
    plot_static(poro, perm, channels)
    plot_data(timestamps, well_opr, well_wpr, well_wcut)
    plot_dynamic(poro, saturation, multiplier=10, cmaps=['viridis','jet']) 
    # make_dynamic_animation(poro, [pressure,saturation])
    
    ### Process data
    # X_data, y_data, w_data = split_xyw(poro,perm,channels,pressure,saturation,well_opr,well_wpr,well_wcut)
    X_data, y_data, w_data, timestamps = load_xywt()
    xarr, yarr, warr, locs = my_train_test_split(X_data, y_data, w_data, nobs=25, equigrid=False)
    [X_train,X_test], [y_train,y_test], [w_train,w_test], [randx,randy] = xarr, yarr, warr, locs
    plot_X_img_observation(X_data, randx, randy)
    plot_X_observation(X_train)
    plot_X_line_observation(X_train, timestamps)
    
    # AUTOENCODER models
    static_enc,  static_dec,  static_ae,  static_fit  = make_static_ae(y_train)
    dynamic_enc, dynamic_dec, dynamic_ae, dynamic_fit = make_dynamic_ae(X_train)
    data_enc,    data_dec,    data_ae,    data_fit    = make_data_ae(w_train)
    plot_loss_all(data_fit, static_fit, dynamic_fit)

    y_train_pred, y_test_pred = make_ae_prediction(y_train, y_test, static_ae)
    plot_static_results([y_train, y_test], [y_train_pred, y_test_pred], train_or_test='train', multiplier=10, channel_select=0)
    plot_static_results([y_train, y_test], [y_train_pred, y_test_pred], train_or_test='test', multiplier=10, channel_select=0)

    w_train_pred, w_test_pred = make_ae_prediction(w_train, w_test, data_ae)
    plot_data_results(timestamps, w_train, w_test, w_train_pred, w_test_pred, channel_select=2, multiplier=10)
        
    X_train_pred, X_test_pred = make_ae_prediction(X_train, X_test, dynamic_ae)
    plot_dynamic_results([X_train, X_test], [X_train_pred, X_test_pred], train_or_test='train', multiplier=10)
    plot_dynamic_results([X_train, X_test], [X_train_pred, X_test_pred], train_or_test='test', multiplier=10)

    # Join train-test data for full transfer learning
    X_full, w_full, y_full = make_full_traintest(X_train, X_test, w_train, w_test, y_train, y_test)

    # Latent INVERSE model
    inv, inv_fit = make_inv_regressor(X_full, w_full, y_full, dynamic_enc, data_enc, static_dec)
    plot_loss(inv_fit, figsize=(4,3), savefig=True, fname='inverse_reg_trainingperformance')
        
    inv_train, inv_test = make_inv_prediction(inv, [X_train, X_test], [w_train, w_test], [y_train, y_test])
    plot_inv_latent_dashboard(X_full, w_full, inv_train, realization=sample_realization)

    plot_inversion_result([y_train, y_test], [inv_train, inv_test], train_or_test='train', channel_select=0)
    plot_inversion_result([y_train, y_test], [inv_train, inv_test], train_or_test='train', channel_select=0)

    poro_hat, perm_hat, channels_hat = make_inv_backnorm([inv_train, inv_test], [poro, perm, channels])
    plot_static(poro_hat, perm_hat, channels_hat, multiplier=10, inv_flag=True)

    # Latent FORWARD model
    fwd, fwd_fit = make_fwd_regressor(X_full, w_full, y_full, dynamic_dec, data_dec, static_enc)
    plot_loss(fwd_fit, figsize=(4,3), savefig=True, fname='forward_reg_trainingperformance')
        
    fwd_X, fwd_w = make_fwd_prediction(fwd, [X_train, X_test], [w_train, w_test], [y_train, y_test])
    plot_fwd_latent_dashboard(w_full, y_full, fwd_X['train'], fwd_w['train'])

    plot_fwd_results_data([w_train,w_test], fwd_w, train_or_test='train')
    plot_fwd_results_data([w_train,w_test], fwd_w, train_or_test='test')

    plot_fwd_results_dynamic([X_train,X_test], fwd_X, train_or_test='train')
    plot_fwd_results_dynamic([X_train,X_test], fwd_X, train_or_test='test')

    plot_fwd_results_dynamic_line([X_train,X_test], fwd_X, train_or_test='train')
    plot_fwd_results_dynamic_line([X_train,X_test], fwd_X, train_or_test='test')
    
    spatial_fwdX = fwd_latent_spatial_interpolation(fwd_X, locs=[randx,randy], train_or_test='train', n_obs=100)
    plot_fwd_latent_map(spatial_fwdX, y_data, X_data, locs=[randx, randy], realization=sample_realization)
    xcum_true, xcum_pred, uq = plot_fwd_resoil(X_data, spatial_fwdX, correction=0.75, cmap='rainbow')

    # Save models
    print('-------------------------------------')
    print('Saving models to disk...')
    save_models([static_enc, dynamic_enc, data_enc],
                [static_dec, dynamic_dec, data_dec],
                [static_ae,  dynamic_ae,  data_ae],
                [inv, fwd])

    # Exit
    print('All done! | Total time: {:.2f} minutes'.format((time()-time0)/60))
    print('-------------------------------------')

################################################################################################
############################################# END ##############################################
################################################################################################