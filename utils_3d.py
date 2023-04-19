################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import pyvista as pv

from scipy.io import loadmat
from time import time

from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity as img_ssim

import keras.backend as K
from keras import Model, Input
from tensorflow_addons.layers import InstanceNormalization, GELU
from keras.layers import BatchNormalization, LayerNormalization, PReLU, LeakyReLU
from keras.layers import Flatten, Reshape, Concatenate, Lambda
from keras.layers import SeparableConv2D, AveragePooling2D, UpSampling2D, Dense, Dropout
from keras.optimizers import Adam, Nadam
from keras.losses import mean_squared_error as loss_mse
from tensorflow.image import ssim as loss_ssim
################################################################################################

n_realizations, n_timesteps = 318, 40
xy_dim, z_depth, n_wells    = 48, 8, 9
static_channels, dynamic_channels, data_channels = 3, 2, 4

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

def my_normalize(data, scaler=None, mode='forward', feature='static', data_orig=None):
    if mode=='forward':
        if feature=='dynamic':
            feature_min = data.min(axis=(-3,-2), keepdims=True)
            feature_max = data.max(axis=(-3,-2), keepdims=True)
            return (data-feature_min)/(feature_max-feature_min)
        elif feature=='static':
            scaler = MinMaxScaler()
            scaler.fit(data.reshape(n_realizations,-1))
            data_norm_f = scaler.transform(data.reshape(n_realizations,-1))
            data_norm = np.reshape(data_norm_f, data.shape)
            return data_norm, scaler
        elif feature=='data':
            data_norm = np.zeros(data.shape)
            for i in range(data.shape[-1]):
                data_min = data[...,i].min(axis=(-2,-1), keepdims=True)
                data_max = data[...,i].max(axis=(-2,-1), keepdims=True)
                data_norm[...,i] = (data[...,i]-data_min)/(data_max-data_min)
            return data_norm
        else: 
            print('Select feature type: [static, dynamic, data]')
    elif mode=='inverse':
        if feature=='dynamic':
            feature_min = data_orig.min(axis=(-3,-2), keepdims=True)
            feature_max = data_orig.max(axis=(-3,-2), keepdims=True)
            return data*(feature_max-feature_min)+feature_min
        elif feature=='static':
            data_inv_f = scaler.inverse_transform(data.reshape(318,-1))
            return data_inv_f.reshape(data.shape)
        elif feature=='data':
            data_inv = np.zeros(data.shape)
            for i in range(data.shape[-1]):
                true_max = data_orig[...,i].max(axis=(-2,-1), keepdims=True)
                true_min = data_orig[...,i].min(axis=(-2,-1), keepdims=True)
                data_inv[...,i] = data[...,i]*(true_max-true_min)+true_min
            return data_inv
        else: 
            print('Select feature type: [static, dynamic, data]')
    else:
        print('Select normalization mode: [forward, inverse]')
        
def make_initial_data(n_realizations=n_realizations, save=True):
    satu = np.zeros((n_realizations,n_timesteps,xy_dim,xy_dim,8))
    pres = np.zeros((n_realizations,n_timesteps,xy_dim,xy_dim,8))
    poro = np.zeros((n_realizations,xy_dim,xy_dim,8))
    perm = np.zeros((n_realizations,xy_dim,xy_dim,8))
    prod = np.zeros((n_realizations,n_timesteps,9,4))
    for i in range(n_realizations):
        k = i+1
        satu[i] = np.moveaxis(loadmat('E:/Latent_Geo_Inversion/simulations_3D/saturation/saturation_{}.mat'.format(k))['satu'].reshape(n_timesteps,z_depth,xy_dim,xy_dim).T, -1, 0)
        pres[i] = np.moveaxis(loadmat('E:/Latent_Geo_Inversion/simulations_3D/pressure/pressure_{}.mat'.format(k))['pres'].reshape(n_timesteps,z_depth,xy_dim,xy_dim).T, -1, 0)/10
        poro[i] = loadmat('E:/Latent_Geo_Inversion/simulations_3D/porosity/porosity_{}.mat'.format(k))['porosity'].reshape(z_depth,xy_dim,xy_dim).T
        perm[i] = loadmat('E:/Latent_Geo_Inversion/simulations_3D/permeability/permeability_{}.mat'.format(k))['perm_md'].reshape(z_depth,xy_dim,xy_dim).T
        prod[i] = loadmat('E:/Latent_Geo_Inversion/simulations_3D/production/production_{}'.format(k))['production']
    prod[:,:,:,0] /= 75
    facies = np.load('simulations 3D/facies_maps_48_48_8.npy')
    timestamps = loadmat('simulations 3D/timestamp_yr.mat')['timestamps_yr'].squeeze()
    print('Pres: {} | Satu: {}\nPoro: {} | Perm: {} | Facies: {}'.format(pres.shape, satu.shape, poro.shape, perm.shape, facies.shape))
    print('Timestamps: {} | Production: {}'.format(timestamps.shape, prod.shape))
    if save:
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/saturation.npy', satu)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/pressure.npy', pres)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/porosity.npy', poro)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/permeability.npy', perm)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/facies.npy', facies)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/production.npy', prod)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/timestamps.npy', timestamps)
    return satu, pres, poro, perm, facies, prod, timestamps

def load_initial_data():
    satu = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/saturation.npy')
    pres = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/pressure.npy')
    poro = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/porosity.npy')
    perm = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/permeability.npy')
    facies = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/facies.npy')
    prod = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/production.npy')
    timestamps = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/timestamps.npy')
    print('Pres: {} | Satu: {}\nPoro: {} | Perm: {} | Facies: {}'.format(pres.shape, satu.shape, poro.shape, perm.shape, facies.shape))
    print('Timestamps: {} | Production: {}'.format(timestamps.shape, prod.shape))
    return satu, pres, poro, perm, facies, prod, timestamps

def split_xywt(facies, poro, perm, pres, satu, prod, timestamps, save=False):
    y_data = np.zeros((n_realizations,xy_dim,xy_dim,z_depth,static_channels))
    y_data[...,0] = my_normalize(facies, feature='static')[0]
    y_data[...,1] = my_normalize(poro, feature='static')[0]
    y_data[...,2] = my_normalize(np.log10(perm), feature='static')[0]
    X_data = np.zeros((n_realizations,n_timesteps,xy_dim,xy_dim,z_depth,dynamic_channels))
    X_data[...,0] = my_normalize(pres, feature='dynamic')
    X_data[...,1] = satu
    w_data = my_normalize(prod, feature='data')
    if save:
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/X_data.npy', X_data)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/y_data.npy', y_data)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/w_data.npy', w_data)
        np.save('E:/Latent_Geo_Inversion/simulations_3D/data/t_data.npy', timestamps)
    print('X shape: {} | y shape: {} | w shape: {}\nt shape: {}'.format(X_data.shape, y_data.shape, prod.shape, timestamps.shape))
    return X_data, y_data, w_data, timestamps

def load_xywt():
    x = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/X_data.npy')
    y = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/y_data.npy')
    w = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/w_data.npy')
    t = np.load('E:/Latent_Geo_Inversion/simulations_3D/data/t_data.npy')
    print('X shape: {} | y shape: {} | w shape: {}\nt shape: {}'.format(x.shape, y.shape, w.shape, t.shape))
    return x, y, w, t

def my_train_test_split(X, y, w, n_train=250, n_obs=30):
    def reshape_y(data3d, len_tr_or_te):
        y0 = np.moveaxis(data3d, -2, 1).reshape(len_tr_or_te*z_depth,xy_dim,xy_dim,static_channels)
        return y0
    def reshape_X(data4d, len_tr_or_te):
        x0 = np.moveaxis(data4d, -2, 1).reshape(len_tr_or_te*z_depth,n_timesteps,n_obs,dynamic_channels)
        return x0
    def reshape_w(data2d, len_tr_or_te):
        w1 = np.moveaxis(np.repeat(np.expand_dims(data2d,-1), z_depth, -1),-1,1)
        w0 = w1.reshape(len_tr_or_te*z_depth,n_timesteps,n_wells,data_channels)
        return w0
    train_idx = np.random.choice(np.arange(n_realizations), n_train, replace=False)
    test_idx  = np.setdiff1d(np.arange(n_realizations), train_idx)
    randx, randy = np.random.randint(xy_dim, size=n_obs), np.random.randint(xy_dim, size=n_obs)
    idxs, rands = [train_idx, test_idx], [randx, randy]
    n_train, n_test = len(train_idx), len(test_idx)
    X_train, X_test = reshape_X(X[train_idx][:,:,randx,randy], n_train), reshape_X(X[test_idx][:,:,randx,randy], n_test)
    y_train, y_test = reshape_y(y[train_idx], n_train), reshape_y(y[test_idx], n_test)
    w_train, w_test = reshape_w(w[train_idx], n_train), reshape_w(w[test_idx], n_test)
    print('X_train shape: {} | X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('w_train shape: {}  | w_test shape: {}'.format(w_train.shape, w_test.shape))
    print('y_train shape: {} | y_test shape: {}'.format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, w_train, w_test, idxs, rands

################################################################################################
def plot_data(timestamps, production, multiplier=1, ncols=10, figsize=(25,8)):
    labels = ['BHP [psia]', 'Oil rate [stb/d]', 'Water rate [stb/d]', 'Water Cut [v/v]']
    well_names = ['I1','I2','I3','I4','I5','P1','P2','P3','P4']
    fig, axs = plt.subplots(data_channels, ncols, figsize=figsize)
    for i in range(data_channels):
        for j in range(ncols):
            axs[i,j].plot(timestamps, production[j*multiplier,:,:,i])
            axs[0,j].set(title='Realization {}'.format(j))
            axs[i,j].grid('on')
        axs[i,0].set(ylabel=labels[i])
        fig.legend(labels=well_names, loc='right', bbox_to_anchor=(0.95, 0.5))   

def plot_static(facies, poro, perm, multiplier=1, ncols=10, windowsize=(1500,600), cmaps=['viridis','jet','jet']):
    data = [facies, poro, np.log10(perm)]
    labels = ['Facies', 'Porosity', 'Log-Perm']
    p = pv.Plotter(shape=(len(data),ncols))
    for j in range(ncols):
        for i in range(len(data)):
            cb_args = {'title':labels[i], 'n_labels':3, 'fmt':'%.1f',
                       'title_font_size':15, 'label_font_size':8}
            p.subplot(i,j)
            p.add_mesh(np.flip(data[i][j*multiplier]), cmap=cmaps[i], scalar_bar_args=cb_args)
        p.subplot(0,j); p.add_title('Realization {}'.format(j*multiplier), font_size=6)
    p.show(jupyter_backend='static', window_size=windowsize)

def plot_dynamic(static, dynamic, nrows=5, multiplier=1, windowsize=(1500,800), cmaps=['viridis', 'jet']):
    times = [0, 4, 9, 14, 19, 24, 29, 34, 39]
    p = pv.Plotter(shape=(nrows, len(times)))
    for i in range(nrows):
        p.subplot(i,0)
        cb_s_args = {'title':'Static','n_labels':3,'fmt':'%.1f','title_font_size':15,'label_font_size':8}
        p.add_mesh(np.flip(static[i*multiplier]), cmap=cmaps[0], scalar_bar_args=cb_s_args)
        p.add_title('Realization {}'.format(i*multiplier), font_size=8)
        for j in range(1,len(times)):
            cb_d_args = {'title':'Dynamic','n_labels':3,'fmt':'%.1f','title_font_size':15,'label_font_size':8}
            p.subplot(i,j)
            p.add_mesh(dynamic[i,times[j]], cmap=cmaps[1], scalar_bar_args=cb_d_args)
            p.add_title('step {}'.format(times[j]+1), font_size=8)
    p.show(jupyter_backend='satic', window_size=windowsize)

def plot_X_observations(data, ncols=10, multiplier=1, figsize=(20,3), cmaps=['gnuplot2','jet']):
    fig, axs = plt.subplots(dynamic_channels, ncols, figsize=figsize, sharex=True, sharey=True)
    n_samples, n_obs = int(data.shape[0]/z_depth), int(data.shape[-2])
    df = data.reshape(n_samples, z_depth, n_timesteps, n_obs, dynamic_channels)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].imshow(df[k,0,:,:,0].T, cmap=cmaps[0])
        axs[1,i].imshow(df[k,0,:,:,1].T, cmap=cmaps[1])
        axs[0,i].set(title='Realization {}'.format(k))
    axs[0,i].set_ylabel('Pressure', labelpad=-110, rotation=270)
    axs[1,i].set_ylabel('Saturation', labelpad=-110, rotation=270)
    fig.text(0.5, 0.01, 'Timesteps', ha='center')
    fig.text(0.1, 0.5, 'Location Index', va='center', rotation='vertical')

def plot_X_line_observations(data, times, ncols=10, multiplier=1, figsize=(20,5)):
    fig, axs = plt.subplots(dynamic_channels, ncols, figsize=figsize, sharex=True, sharey=True)
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

def plot_X_img_observations(data, rands, timing=-1, multiplier=1, ncols=10, figsize=(20,4), cmaps=['gnuplot2', 'jet']):
    randx, randy = rands
    fig, axs = plt.subplots(dynamic_channels, ncols, figsize=figsize)
    for i in range(ncols):
        k = i*multiplier
        axs[0,i].imshow(data[k,timing,:,:,0,0], cmap=cmaps[0])
        axs[0,i].scatter(randx, randy, marker='s', c='k')
        axs[1,i].imshow(data[k,timing,:,:,0,1], cmap=cmaps[1])
        axs[1,i].scatter(randx, randy, marker='s', c='k')
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(dynamic_channels):
            axs[j,i].set(xticks=[], yticks=[])
    axs[0,0].set(ylabel='Pressure')
    axs[1,0].set(ylabel='Saturation')

def plot_loss(fit, title='', figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    loss, val  = fit.history['loss'], fit.history['val_loss']
    epochs     = len(loss)
    iterations = np.arange(epochs)
    plt.plot(iterations, loss, '-', label='loss')
    plt.plot(iterations, val, '-', label='validation loss')
    plt.title(title+' Training: Loss vs epochs'); plt.legend()
    plt.ylabel('Epochs'); plt.ylabel('Loss')
    plt.xticks(iterations[::epochs//10])

def plot_loss_all(loss1, loss2, loss3, titles=['Data','Static','Dynamic'], figsize=(15,3)):
    plt.figure(figsize=figsize, facecolor='white')
    losses = [loss1, loss2, loss3]
    for i in range(len(losses)):
        plt.subplot(1,len(losses),i+1)
        plot_loss(losses[i], title=titles[i])

def plot_data_results(timestamps, true, pred, ncols=10, multiplier=1, figsize=(20,8), suptitle='___'):
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan']
    labels = ['BHP [psia]', 'Oil Rate [stb/d]', 'Water Rate [stb/d]', 'Water Cut [%]']
    fig, axs = plt.subplots(data_channels, ncols, figsize=figsize, facecolor='white')
    n_samples = int(true.shape[0]/z_depth)
    truth = true.reshape(n_samples,z_depth,len(timestamps),n_wells,data_channels) 
    hat   = pred.reshape(n_samples,z_depth,len(timestamps),n_wells,data_channels)
    for i in range(data_channels):
        for j in range(ncols):
            for k in range(5):
                axs[i,j].plot(timestamps, truth[j*multiplier,0,:,k,i], label='I{} true'.format(k+1), c=colors[k], linestyle='-')
                axs[i,j].plot(timestamps, hat[j*multiplier,0,:,k,i],   label='I{} pred'.format(k+1), c=colors[k], linestyle='--')
            for m in range(5,9):
                axs[i,j].plot(timestamps, truth[j*multiplier,0,:,m,i], label='P{} true'.format(m-4), c=colors[m], linestyle='-')
                axs[i,j].plot(timestamps, hat[j*multiplier,0,:,m,i],   label='P{} pred'.format(m-4), c=colors[m], linestyle='--')
            axs[0,j].set(title='Realization {}'.format(j*multiplier))
        axs[i,0].set(ylabel=labels[i])
    for j in range(1,ncols):
        for i in range(data_channels):
            axs[i,j].set(yticks=[])
    for i in range(3):
        for j in range(ncols):
            axs[i,j].set(xticks=[])
    fig.text(0.5, 0.04, 'Time [years]', ha='center')
    plt.suptitle(suptitle + ' Observations')
    plt.legend(bbox_to_anchor=(2, 4))

def plot_static_results(true, pred, channel_select=0, ncols=10, multiplier=1, cmaps=['jet','seismic'], windowsize=(1500,800)):
    n_samples = int(true.shape[0]/z_depth)
    truth = np.moveaxis(true.reshape(n_samples,z_depth,xy_dim,xy_dim,static_channels), 1, -2)
    hat   = np.moveaxis(pred.reshape(n_samples,z_depth,xy_dim,xy_dim,static_channels), 1, -2)
    labels = ['True', 'Prediction', 'Difference']
    fcmap  = [cmaps[0], cmaps[0], cmaps[1]]
    p = pv.Plotter(shape=(3,ncols))
    for j in range(ncols):
        true_vol = np.flip(truth[j*multiplier,:,:,:,channel_select])
        pred_vol = np.flip(hat[j*multiplier,:,:,:,channel_select])
        diff_vol = true_vol - pred_vol
        vols = [true_vol, pred_vol, diff_vol]
        for i in range(3):
            cb_args = {'title':labels[i], 'n_labels':3, 'fmt':'%.1f', 
                        'title_font_size':15, 'label_font_size':8}
            p.subplot(i,j); p.add_mesh(vols[i], cmap=fcmap[i], scalar_bar_args=cb_args)
            p.subplot(0,j); p.add_title('Realization {}'.format(j*multiplier), font_size=6)
    p.show(jupyter_backend='static', window_size=windowsize)

def plot_dynamic_results(true, pred, channel_select=0, ncols=10, multiplier=1, figsize=(20,4.5), suptitle='___'):
    fig, axs = plt.subplots(3, ncols, figsize=figsize, sharex=True, sharey=True)
    n_samples, n_obs = int(true.shape[0]/z_depth), int(true.shape[-2])
    df_true = true.reshape(n_samples,z_depth,n_timesteps,n_obs,dynamic_channels)
    df_pred = pred.reshape(n_samples,z_depth,n_timesteps,n_obs,dynamic_channels)
    if channel_select==0:
        supertitle, fcmap = 'Pressure', 'turbo'
    elif channel_select==1:
        supertitle, fcmap = 'Saturation', 'jet'
    else:
        print('Select dynamic channel [0=Pressure, 1=Saturation]')
        return None
    fcmap, labels = [fcmap,fcmap,'seismic'], ['True','Prediction','Difference']
    for i in range(3):
        for j in range(ncols):
            k = j*multiplier
            real = df_true[k,0,:,:,channel_select].T
            hat  = df_pred[k,0,:,:,channel_select].T
            diff = real - hat
            imgs = [real, hat, diff]
            axs[0,j].set(title='Realization {}'.format(k))
            im = axs[i,j].imshow(imgs[i], cmap=fcmap[i])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        axs[i,0].set(ylabel=labels[i])
    plt.suptitle(suptitle + ' ' + supertitle)
    fig.text(0.5, 0.01, 'Timestep [years]', ha='center')

################################################################################################
def conv_block(inp, filt, kern=(3,3), pool=(2,2), pad='same'):
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(_)
    _ = InstanceNormalization()(_)
    #_ = BatchNormalization()(_)
    _ = GELU()(_)
    _ = AveragePooling2D(pool)(_)
    return _

def decon_block(inp, filt, kern=(3,3), pool=(2,2), pad='same'):
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(inp)
    _ = SeparableConv2D(filters=filt, kernel_size=kern, padding=pad)(_)
    _ = InstanceNormalization()(_)
    #_ = BatchNormalization()(_)
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
                    verbose=0, validation_split=0.2, shuffle=True)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(wparams,traintime))
    return enc, dec, vae, fit

def make_static_ae(y, epochs=600, batch=50, opt=Adam(1e-3), ssim_perc=(2/3)):
    input_static = Input(shape=y.shape[1:])
    _ = conv_block(input_static, 8)
    _ = conv_block(_, 16)
    _ = conv_block(_, 32)
    code = SeparableConv2D(64, (3,3), padding='same', activation='relu')(_)
    shape_b4 = K.int_shape(code)[1:]
    latent = Flatten()(code)
    shape_flat = K.int_shape(latent)[1]
    z_inp = Input(shape=(shape_flat,))
    _ = Reshape(shape_b4)(z_inp)
    _ = decon_block(_, 32)
    _ = decon_block(_, 16)
    _ = decon_block(_, 8)
    output = SeparableConv2D(3, (3,3), padding='same', activation='sigmoid')(_)
    enc = Model(input_static, latent, name='static_encoder')
    dec = Model(z_inp, output, name='static_decoder')
    output_static = dec(enc(input_static))
    ae = Model(input_static, output_static, name='static_ae')
    ssim = 1 - tf.reduce_mean(loss_ssim(input_static, output_static, 1.0))
    mse = loss_mse(input_static, output_static)
    dual_loss = (ssim_perc)*ssim + (1-ssim_perc)*mse
    ae.add_loss(dual_loss)
    ae.compile(optimizer=opt, metrics=['mse'])
    yparams = ae.count_params()
    start = time()
    fit = ae.fit(y, y, epochs=epochs, batch_size=batch, 
                    verbose=0, validation_split=0.2, shuffle=True)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(yparams,traintime))
    return enc, dec, ae, fit

def make_dynamic_ae(x, code_dim=1000, z_dim=20, epochs=100, batch=50, opt=Adam(1e-3)):
    def sample(args, mu=0.0, std=1.0):
        mean, sigma = args
        epsilon = K.random_normal(shape=(K.shape(mean)[0],z_dim), mean=mu, stddev=std)
        return mean + K.exp(sigma)*epsilon
    def dense_block(inp, units, drop=0.2):
        _ = Dense(units)(inp)
        _ = BatchNormalization()(_)
        _ = PReLU()(_)
        _ = Dropout(drop)(_)
        return _
    inputs = Input(shape=x.shape[1:])
    shape_b4 = K.int_shape(inputs)[1:]
    _ = Flatten()(inputs)
    _ = dense_block(_, code_dim)
    code = _
    _ = dense_block(_, 100)
    mean = Dense(z_dim)(_)
    sigma = Dense(z_dim)(_)
    latent = Lambda(sample)([mean, sigma])
    z_inp = Input(shape=(z_dim,))
    _ = dense_block(z_inp, 100)
    _ = dense_block(_, code_dim)
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
                    verbose=0, validation_split=0.2, shuffle=True)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(xparams,traintime))
    return enc, dec, vae, fit

def make_ae_prediction(train_true, test_true, ae_model):
    train_pred = ae_model.predict(train_true).astype('float64')
    test_pred  = ae_model.predict(test_true).astype('float64')
    mse_train, mse_test = img_mse(train_true, train_pred), img_mse(train_true, train_pred)
    print('Train MSE: {:.2e} | Test MSE: {:.2e}'.format(mse_train, mse_test))
    if train_true.shape[2]>=7:
        ssim_train = img_ssim(train_true, train_pred, channel_axis=-1)
        ssim_test  = img_ssim(test_true, test_pred, channel_axis=-1)
        print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    else:
        print('Image data must have shape at least (7x7) for ssim calculation')
    return train_pred, test_pred

def make_full_traintest(xtrain, xtest, wtrain, wtest, ytrain, ytest):
    n_train, n_test = int(xtrain.shape[0]/z_depth), int(xtest.shape[0]/z_depth)
    n_obs = int(xtrain.shape[-2])
    xtr0 = xtrain.reshape(n_train,z_depth,n_timesteps,n_obs,dynamic_channels)
    xte0 = xtest.reshape(n_test,z_depth,n_timesteps,n_obs,dynamic_channels)
    xfull = np.concatenate([xtr0, xte0]).reshape(n_realizations*z_depth,n_timesteps,n_obs,dynamic_channels)
    ytr0 = ytrain.reshape(n_train,z_depth,xy_dim,xy_dim,static_channels)
    yte0 = ytest.reshape(n_test,z_depth,xy_dim,xy_dim,static_channels)
    yfull = np.concatenate([ytr0,yte0]).reshape(n_realizations*z_depth,xy_dim,xy_dim,static_channels)
    wtr0 = wtrain.reshape(n_train,z_depth,n_timesteps,n_wells,data_channels)
    wte0 = wtest.reshape(n_test,z_depth,n_timesteps,n_wells,data_channels)
    wfull = np.concatenate([wtr0,wte0]).reshape(n_realizations*z_depth,n_timesteps,n_wells,data_channels)
    print('X_full: {} | w_full: {} | y_full: {}'.format(xfull.shape, wfull.shape, yfull.shape))
    return xfull, wfull, yfull

def make_inv_regressor(xf, wf, yf, dynamic_enc, data_enc, static_dec, 
                            opt=Nadam(1e-4), loss='mse', epochs=600, batch=80):
    dynamic_enc.trainable = False
    data_enc.trainable    = False
    static_dec.trainable  = False
    def dense_block(input, neurons):
        _ = Dense(neurons, kernel_regularizer='l1')(input)
        _ = LayerNormalization()(_)
        #_ = BatchNormalization()(_)
        _ = LeakyReLU()(_)
        return _
    x_inp = Input(shape=xf.shape[1:])
    x_latent = dynamic_enc(x_inp)[-1]
    #x = dense_block(x_latent, 1000)
    #x = dense_block(x, 2000)
    w_inp = Input(shape=wf.shape[1:])
    w_latent = data_enc(w_inp)[-1]
    #w = dense_block(w_latent, 300)
    #w = dense_block(w, 600)
    #w = dense_block(w, 1000)
    _ = Concatenate()([x_latent, w_latent])
    _ = LayerNormalization()(_)
    #_ = dense_block(_, 2000)
    _ = Dense(6*6*64)(_)
    out = static_dec(_)
    reg = Model([x_inp, w_inp], out)
    rparams = reg.count_params()
    reg.compile(optimizer=opt, loss=loss, metrics=['mse'])
    start = time()
    fit = reg.fit([xf, wf], yf, epochs=epochs, batch_size=batch, 
                    verbose=0, validation_split=0.2, shuffle=True)
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
    ssim_train = img_ssim(ytrain, inv_train, channel_axis=-1)
    ssim_test  = img_ssim(ytest, inv_test, channel_axis=-1)
    print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    return inv_train, inv_test

def make_inv_backnorm(data_inv, data_orig, idxs):
    inv_train, inv_test = data_inv
    facies0, poro0, perm0 = data_orig
    data0 = np.concatenate([np.expand_dims(facies0,-1), np.expand_dims(poro0,-1), np.expand_dims(perm0,-1)], -1)
    n_train, n_test = int(inv_train.shape[0]/z_depth), int(inv_test.shape[0]/z_depth)
    inv_tr0 = inv_train.reshape(n_train,z_depth,xy_dim,xy_dim,static_channels)
    inv_te0 = inv_test.reshape(n_test,z_depth,xy_dim,xy_dim,static_channels)
    new_pred = np.moveaxis(np.concatenate([inv_tr0,inv_te0]),1,-2)
    new_true = np.take(data0, np.concatenate([idxs[0],idxs[1]]), axis=0)
    facies_scaler = my_normalize(new_true[...,0], mode='forward', feature='static')[1]
    poro_scaler   = my_normalize(new_true[...,1], mode='forward', feature='static')[1]
    perm_scaler   = my_normalize(new_true[...,2], mode='forward', feature='static')[1]
    facies_hat = my_normalize(new_pred[...,0], scaler=facies_scaler, mode='inverse', data_orig=new_true[...,0])
    poro_hat   = my_normalize(new_pred[...,1], scaler=poro_scaler, mode='inverse', data_orig=new_true[...,1])
    perm_hat   = my_normalize(new_pred[...,2], scaler=perm_scaler, mode='inverse', data_orig=new_true[...,2])
    return facies_hat, poro_hat, perm_hat