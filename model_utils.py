### Load main packages ######################################################
import numpy as np
from time import time
import tensorflow as tf

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

### define convolution and deconvolution blocks #############################
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

### Make Data, Static, Dynamic autoencoders #################################
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

### generate inversion predictions ##########################################
def make_inv_prediction(regmodel, x_tuple, w_tuple, ytrain, ytest):
    xtrain, xtest = x_tuple
    wtrain, wtest = w_tuple
    inv_train = regmodel.predict([xtrain, wtrain]).astype('float64')
    inv_test = regmodel.predict([xtest, wtest]).astype('float64')
    mse_train, mse_test = img_mse(ytrain, inv_train), img_mse(ytest, inv_test)
    print('Train MSE: {:.2e} | Test MSE: {:.2e}'.format(mse_train, mse_test))
    ssim_train = structural_similarity(ytrain, inv_train, channel_axis=-1)
    ssim_test  = structural_similarity(ytest, inv_test, channel_axis=-1)
    print('Train SSIM: {:.2f} | Test SSIM: {:.2f}'.format(100*ssim_train, 100*ssim_test))
    return inv_train, inv_test