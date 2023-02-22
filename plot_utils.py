import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### Loss functions
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

### Visualize original data individiually
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

### Engineering dynamic observations (observation wells)
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

### truth vs. predicted results
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