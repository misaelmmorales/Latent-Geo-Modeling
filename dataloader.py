import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat

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

def make_arrays():
    well_opr   = np.zeros((n_realizations, n_timesteps, 3))
    well_wpr   = np.zeros((n_realizations, n_timesteps, 3))
    well_wcut  = np.zeros((n_realizations, n_timesteps, 3))
    perm       = np.zeros(shape=(n_realizations, xy_dim, xy_dim))
    poro       = np.zeros(shape=(n_realizations, xy_dim, xy_dim))
    pressure   = np.zeros(shape=(n_realizations, n_timesteps, xy_dim, xy_dim))
    saturation = np.zeros(shape=(n_realizations, n_timesteps, xy_dim, xy_dim))
    
    timestamps = loadmat('simulations/response_production/production_1.mat')['Prod']['t'][0][0].flatten()[1:]*3.1689E-8
    channels = np.transpose(np.array(pd.read_csv('simulations/channel_all.csv', header=None)).T.reshape(n_realizations, xy_dim,xy_dim), axes=(0,2,1))
    for i in range(n_realizations):
        well_opr[i] = loadmat('simulations/response_production/production_{}.mat'.format(i+1))['Prod']['opr'][0][0][1:, 3:]*5.4344E5 #to bbls
        well_wpr[i] = loadmat('simulations/response_production/production_{}.mat'.format(i+1))['Prod']['wpr'][0][0][1:, 3:]*5.4344E5 #to bbls
        well_wcut[i] = loadmat('simulations/response_production/production_{}.mat'.format(i+1))['Prod']['wc'][0][0][1:, 3:]
    for i in range(n_realizations):
        poro[i,:,:] = loadmat('simulations/features_porosity/porosity_{}.mat'.format(i+1))['poro'].flatten().reshape(xy_dim,xy_dim)
        perm[i,:,:] = np.log10(loadmat('simulations/features_permeability/permeability_{}.mat'.format(i+1))['permeability']).flatten().reshape(xy_dim,xy_dim)
        pressure[i,:,:,:]   = loadmat('simulations/response_pressure/pressure_{}.mat'.format(i+1))['pres'].T.reshape(n_timesteps,xy_dim,xy_dim)**0.00689476 #to psi
        saturation[i,:,:,:] = loadmat('simulations/response_saturation/saturation_{}.mat'.format(i+1))['satu'].T.reshape(n_timesteps,xy_dim,xy_dim)

    np.save('data/well_opr.npy', well_opr); np.save('data/well_wpr.npy', well_wpr); np.save('data/well_wcut.npy', well_wcut)
    np.save('data/poro.npy', poro); np.save('data/perm.npy', perm); np.save('data/channels.npy', channels)
    np.save('data/pressure.npy', pressure); np.save('data/saturation.npy', saturation); np.save('data/timestamps.npy', timestamps)
    return timestamps, poro, perm, channels, pressure, saturation, well_opr, well_wpr, well_wcut

def load_arrays():
    timestamps           = np.load('data/timestamps.npy')
    poro, perm, channels = np.load('data/poro.npy'), np.load('data/perm.npy'), np.load('data/channels.npy')
    pressure, saturation = np.load('data/pressure.npy'), np.load('data/saturation.npy')
    well_opr, well_wpr, well_wcut = np.load('data/well_opr.npy'), np.load('data/well_wpr.npy'), np.load('data/well_wcut.npy')
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
    print('X shape: {} | y shape: {} | w shape: {}'.format(X_data.shape, w_data.shape, y_data.shape))
    return X_data, y_data, w_data

def load_xywt():
    x = np.load('data/X_data.npy')
    y = np.load('data/y_data.npy')
    w = np.load('data/w_data.npy')
    t = np.load('data/timestamps.npy')
    print('X shape: {} | y shape: {} \nw shape: {} | t shape: {}'.format(x.shape, w.shape, y.shape, t.shape))
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