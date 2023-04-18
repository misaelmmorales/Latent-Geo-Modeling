################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyvista as pv

from scipy.io import loadmat
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity

import keras.backend as K
from keras import Model, Input
from tensorflow_addons.layers import InstanceNormalization, GELU
from keras.layers import BatchNormalization, LayerNormalization, PReLU
from keras.layers import Conv3D, AveragePooling3D, UpSampling3D, LeakyReLU
from keras.layers import Flatten, Reshape, Concatenate, Lambda
from keras.layers import SeparableConv2D, AveragePooling2D, UpSampling2D, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss_mse
################################################################################################

n_realizations = 318
n_timesteps    = 40
xy_dim         = 48

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
        satu[i] = np.moveaxis(loadmat('E:/Latent_Geo_Inversion/simulations_3D/saturation/saturation_{}.mat'.format(k))['satu'].reshape(40,8,48,48).T, -1, 0)
        pres[i] = np.moveaxis(loadmat('E:/Latent_Geo_Inversion/simulations_3D/pressure/pressure_{}.mat'.format(k))['pres'].reshape(40,8,48,48).T, -1, 0)/10
        poro[i] = loadmat('E:/Latent_Geo_Inversion/simulations_3D/porosity/porosity_{}.mat'.format(k))['porosity'].reshape(8,48,48).T
        perm[i] = loadmat('E:/Latent_Geo_Inversion/simulations_3D/permeability/permeability_{}.mat'.format(k))['perm_md'].reshape(8,48,48).T
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
    y_data = np.zeros((n_realizations,xy_dim,xy_dim,8,3))
    y_data[...,0], facies_scaler = my_normalize(facies, feature='static')
    y_data[...,1], poro_scaler   = my_normalize(poro, feature='static')
    y_data[...,2], perm_scaler   = my_normalize(np.log10(perm), feature='static')
    X_data = np.zeros((n_realizations,n_timesteps,xy_dim,xy_dim,8,2))
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
        return np.moveaxis(data3d, -2, 1).reshape(len_tr_or_te*8, xy_dim, xy_dim, 3)
    def reshape_X(data4d, len_tr_or_te):
        return np.moveaxis(data4d, -2, 1).reshape(len_tr_or_te*8, n_timesteps, n_obs, 2)
    def reshape_w(data2d, len_tr_or_te):
        return np.moveaxis(np.repeat(np.expand_dims(data2d,-1), 8, -1),-1,1).reshape(len_tr_or_te*8,n_timesteps,9,4)
    train_idx = np.random.choice(np.arange(n_realizations), n_train, replace=False)
    test_idx  = np.setdiff1d(np.arange(n_realizations), train_idx)
    randx = np.random.randint(xy_dim, size=n_obs)
    randy = np.random.randint(xy_dim, size=n_obs)
    n_train, n_test = len(train_idx), len(test_idx)
    X_train, X_test = reshape_X(X[train_idx][:,:,randx,randy], n_train), reshape_X(X[test_idx][:,:,randx,randy], n_test)
    y_train, y_test = reshape_y(y[train_idx], n_train), reshape_y(y[test_idx], n_test)
    w_train, w_test = reshape_w(w[train_idx], n_train), reshape_w(w[test_idx], n_test)
    print('X_train shape: {} | X_test shape: {}'.format(X_train.shape, X_test.shape))
    print('w_train shape: {}   | w_test shape: {}'.format(w_train.shape, w_test.shape))
    print('y_train shape: {} | y_test shape: {}'.format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, w_train, w_test, randx, randy

################################################################################################
def plot_data(timestamps, production, multiplier=1, ncols=10, figsize=(25,8)):
    labels = ['BHP [psia]', 'Oil rate [stb/d]', 'Water rate [stb/d]', 'Water Cut [v/v]']
    well_names = ['I1','I2','I3','I4','I5','P1','P2','P3','P4']
    fig, axs = plt.subplots(4, ncols, figsize=figsize)
    for i in range(4):
        for j in range(ncols):
            axs[i,j].plot(timestamps, production[j*multiplier,:,:,i])
            axs[0,j].set(title='Realization {}'.format(j))
            axs[i,j].grid('on')
        axs[i,0].set(ylabel=labels[i])
        fig.legend(labels=well_names, loc='right', bbox_to_anchor=(0.95, 0.5))   

def plot_static(facies, poro, perm, multiplier=1, ncols=10, windowsize=(1500,200), cmaps=['jet','jet','jet']):
    p = pv.Plotter(shape=(1,ncols))
    for i in range(ncols):
        p.subplot(0,i)
        p.add_mesh(np.flip(facies[i*multiplier]), cmap=cmaps[0])
    p.show(jupyter_backend='static', window_size=windowsize)
    p = pv.Plotter(shape=(1,ncols))
    for i in range(ncols):
        p.subplot(0,i)
        p.add_mesh(np.flip(poro[i*multiplier]), cmap=cmaps[1])
    p.show(jupyter_backend='static', window_size=windowsize)
    p = pv.Plotter(shape=(1,ncols))
    for i in range(ncols):
        p.subplot(0,i)
        p.add_mesh(np.flip(np.log10(perm[i*multiplier])), cmap=cmaps[2])
    p.show(jupyter_backend='static', window_size=windowsize)

def plot_dynamic(static, dynamic, nrows=5, multiplier=1, windowsize=(1500,800), cmaps=['jet', 'jet']):
    times = [0, 4, 9, 14, 19, 24, 29, 34, 39]
    p = pv.Plotter(shape=(nrows, len(times)))
    for i in range(nrows):
        p.subplot(i,0)
        p.add_mesh(np.flip(static[i*multiplier]), cmap=cmaps[0])
        p.add_title('Realization {}'.format(i*multiplier), font_size=8)
        for j in range(1,len(times)):
            p.subplot(i,j)
            p.add_mesh(dynamic[i,times[j]], cmap=cmaps[1])
            p.add_title('step {}'.format(times[j]+1), font_size=8)
    p.show(jupyter_backend='satic', window_size=windowsize)

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
        axs[0,i].imshow(data[k,timing,:,:,0,0], cmap=cmaps[0])
        axs[0,i].scatter(randx, randy, marker='s', c='k')
        axs[1,i].imshow(data[k,timing,:,:,0,1], cmap=cmaps[1])
        axs[1,i].scatter(randx, randy, marker='s', c='k')
        axs[0,i].set(title='Realization {}'.format(k))
        for j in range(2):
            axs[j,i].set(xticks=[], yticks=[])
    axs[0,0].set(ylabel='Pressure')
    axs[1,0].set(ylabel='Saturation')

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
                            verbose=0, validation_split=0.2, shuffle=True)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(wparams,traintime))
    return enc, dec, vae, fit

def make_static_ae(y, epochs=350, batch=50, opt=Adam(1e-3), ssim_perc=(2/3)):
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
    ssim = 1 - tf.reduce_mean(tf.image.ssim(input_static, output_static, 1.0))
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

def make_dynamic_ae(x, code_dim=1000, z_dim=20, epochs=200, batch=50, opt=Adam(1e-3)):
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
                        verbose=0, validation_split=0.2, shuffle=True)
    traintime = (time()-start)/60
    print('# Parameters: {:,} | Training time: {:.2f} minutes'.format(xparams,traintime))
    return enc, dec, vae, fit

### generate autoencoder predictions ########################################
def make_ae_prediction(train_true, test_true, ae_model):
    train_pred = ae_model.predict(train_true).astype('float64')
    test_pred  = ae_model.predict(test_true).astype('float64')
    mse_train = img_mse(train_true, train_pred)
    mse_test  = img_mse(train_true, train_pred)
    print('Train MSE: {:.2e} | Test MSE: {:.2e}'.format(mse_train, mse_test))
    if train_true.shape[2]>=7:
        ssim_train = structural_similarity(train_true, train_pred, channel_axis=-1)
        ssim_test  = structural_similarity(test_true, test_pred, channel_axis=-1)
        print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    else:
        print('Image data must have shape at least (7x7) for ssim calculation')
    return train_pred, test_pred

### make full train+test dataframes #########################################
def make_full_traintest(xtrain, xtest, wtrain, wtest, ytrain, ytest):
    xfull = np.concatenate([xtrain,xtest])
    wfull = np.concatenate([wtrain,wtest])
    yfull = np.concatenate([ytrain,ytest])
    print('X_full: {} | w_full: {} | y_full: {}'.format(xfull.shape, wfull.shape, yfull.shape))
    return xfull, wfull, yfull

### Make latent space regressor model #######################################
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