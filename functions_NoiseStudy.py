__author__ = 'Silvia E Zieger'
__project__ = 'noise vs resolution'

"""Copyright 2020. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import matplotlib
import matplotlib.pylab as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import LineString
import cv2
import math
import multiprocessing as mp
import seaborn as sns
import pandas as pd
import numpy as np
import random
from lmfit import Model
import scipy.signal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mcerp import *
from uncertainties import *
from uncertainties import unumpy
import h5py
import os
from glob import glob
from PIL import Image
import datetime

# global variables
sns.set(style="darkgrid")
sns.set_context('paper')
col = ['#14274e', '#f6830f', '#bb2205']
mark = ['o', 'd']
fs = 13
depth_lim = dict({'optode1': (-5, 4), 'optode2': (-5, 4)})


# =====================================================================================
def prep_plotting_avSD(error, dfoptode, uncer_op1, uncer_op2):
    if error == 'SD' or error == 'sd':
        df_error = [pd.DataFrame([[i.s for i in dfoptode[en][se]] for se in dfoptode[en].columns],
                                 columns=dfoptode[en].index, index=dfoptode[en].columns).T
                    for en in range(len(dfoptode))]
    else:
        df_error = [uncer_op1['sem'], uncer_op2['sem']]

    return df_error


def prepPlot_optodeSet(o, s, error, dfoptode, uncer_op1, uncer_op2):
    if '1' in o:
        dfop, optode_sem, interpol = dfoptode[0], uncer_op1['sem'], uncer_op1['SD_interpol'][s]
        optode_sd = pd.DataFrame([[i.s for i in dfop[se]] for se in dfop.columns], index=dfop.columns,
                                 columns=dfop.index).T
    else:
        dfop, optode_sem, interpol = dfoptode[1], uncer_op2['sem'], uncer_op2['SD_interpol'][s]
        optode_sd = pd.DataFrame([[i.s for i in dfop[se]] for se in dfop.columns], index=dfop.columns,
                                 columns=dfop.index).T
    if error == 'SD':
        dferr = optode_sd
    else:
        dferr = optode_sem

    return dfop, interpol, dferr


def prepPlot_SVerrprop(error, dop1_value, dop2_value, op1_normSEM, op2_normSEM):
    if error == 'SD' or error == 'sd':
        derror1 = dict(map(lambda s: (s, dop1_value[s][['O2 SD', 'iratio SD']]), dop1_value.keys()))
        derror2 = dict(map(lambda s: (s, dop2_value[s][['O2 SD', 'iratio SD']]), dop2_value.keys()))
    else:
        derror1, derror2 = op1_normSEM, op2_normSEM

    for s in derror1.keys():
        derror1[s].columns, derror2[s].columns = ['O2', 'iratio'], ['O2', 'iratio']

    derror = [derror1, derror2]

    return derror


def prepPlot_SVerrprop_ex(o, s, error, dop1_value=None, dop1_param=None, op1_normSEM=None, f1inter_mc=None,
                          dop2_value=None, dop2_param=None, op2_normSEM=None, f2inter_mc=None):
    if '1' in o:
        ls_df = [dop1_value, dop1_param, op1_normSEM, f1inter_mc]
        if any(i == None for i in ls_df):
            raise ValueError('To plot the example, provide all relevant data! Please check dop_value, dop_param, '
                             ', op_normSEM, and finter_mc')
        dfop, dop_para, df_SEM, finter_mc = dop1_value[s], dop1_param[s], op1_normSEM[s], f1inter_mc[s]
    else:
        ls_df = [dop2_value, dop2_param, op2_normSEM, f2inter_mc]
        if any(i == None for i in ls_df):
            raise ValueError('To plot the example, provide all relevant data! Please check dop_value, dop_param, '
                             ', op_normSEM, and finter_mc')
        dfop, dop_para, df_SEM, finter_mc = dop2_value[s], dop2_param[s], op2_normSEM[s], f2inter_mc[s]

    if error == 'SD' or error == 'sd':
        dferr = dfop[['O2 SD', 'iratio SD']]
    else:
        dferr = pd.concat([df_SEM['O2'], pd.DataFrame([i.s for i in df_SEM['iratio']], index=df_SEM.index)], axis=1)
    dferr.columns = ['O2', 'iratio']

    return dfop, dop_para, df_SEM, dferr, finter_mc


def prepMS_plot(index_lp, dic_micro, offset):
    # microsensor preparation
    df_micro = dic_micro['run1'].set_index('Intensity (mV)')
    df_micro['Depth (mm)'] = df_micro['Depth (µm)'] / 1000  # depth in mm

    # microsensor extension to same depth as selected for the optode
    df_ms = pd.DataFrame([df_micro['Depth (mm)'].index, df_micro['Depth (mm)']], index=['Intensity', 'Depth (mm)']).T

    xnew = np.linspace(1, len(df_ms.index), num=int(len(df_ms.index)))
    df_ms.index = xnew
    df_ms.loc[0, :] = [df_ms['Intensity'].loc[:3].to_numpy().mean(), index_lp[0] * 1.05]
    df_ms = df_ms.sort_index()
    df_ms.loc[xnew[-1] + 1, :] = [df_ms['Intensity'].loc[df_ms.shape[0] - 3:].to_numpy().mean(), index_lp[-1] * 1.05]
    df_ms = df_ms.sort_index()
    df_ms['Depth (mm)'] = [i - offset for i in df_ms['Depth (mm)']]

    return df_ms


def sgolay2d(z, window_size, order, derivative=None):
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial: p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains the exponents of the k-th term. First element of
    # tuple is for x second element for y.
    exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs( np.flipud(z[1:half_size+1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size-1:-1, :]) -band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size+1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)
    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size+1:2*half_size+1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


# ------------------------------------------------------------------------
def plot_optode_avSD_v1(conc, dfoptode, error, col, mark, fs, RoI_op):
    fig2, ax2 = plt.subplots(figsize=(8, 8), nrows=3, ncols=len(RoI_op), sharex=True, sharey=True, frameon=False)
    if len(RoI_op) == 1:
        ax2[0].set_title('Optode with different settings', fontsize=fs * 0.9)
    else:
        for n in range(len(RoI_op)):
            ax2[0][n].set_title('Optode-' + str(n+1), fontsize=fs*0.9)

    # plotting part
    ls_handels = list()
    if len(RoI_op) == 1:
        for en in range(len(dfoptode[0].columns)):
            l = ax2[en].errorbar(conc, [i.n for i in dfoptode[0][en]], error[0][en].values, linestyle='None',
                                 marker=mark[0], fillstyle='none', color=col[en], ms=6, capsize=6, label=en)
            ls_handels.append(l)
    else:
        for o in range(len(RoI_op)):
            for en, s in enumerate(dfoptode[o].columns):
                l = ax2[en][o].errorbar(conc, [i.n for i in dfoptode[o][s]], error[o][s].values, linestyle='None',
                                        marker=mark[0], fillstyle='none', color=col[en], ms=6, capsize=6,
                                        label=s.split('t')[0] + 'tting ' + s.split('t')[-1])
                if o == 1:
                    ls_handels.append(l)

    # legend and axis layout / labelling
    if len(RoI_op) == 1:
        ax2[1].legend(handles=ls_handels, loc="upper left", bbox_to_anchor=[1, 0.9], shadow=True, fancybox=True)
    else:
        ax2[1][len(RoI_op)-1].legend(handles=ls_handels, loc="upper left", bbox_to_anchor=[1, 0.9],  shadow=True,
                                     fancybox=True)

    if len(RoI_op) == 1:
        ax2[0].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # top row
        ax2[1].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # middle row
        ax2[2].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # bottom row
    else:
        for o in range(len(RoI_op)):
            ax2[0][o].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # top row
            ax2[1][o].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # middle row
            ax2[2][o].tick_params(axis='both', which='both', direction='out', labelsize=fs * 0.8) # bottom row

    # x,y label position
    fig2.text(0.5, 0.075, 'O$_2$ concentration [%air]', va='center', ha='center', fontsize=fs * 1.2)
    fig2.text(0.025, 0.55, 'Ratio $R/G$', va='center', ha='center', rotation='vertical', fontsize=fs * 1.2)

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.85, top=0.95)
    plt.show()

    return fig2, ax2


def plot_optode_set(o, s, conc, xinter, dfop, interpol, optode_sem, fs=11):
    fig2, ax2 = plt.subplots(figsize=(5, 3), frameon=False)
    ax2.set_title(o, fontsize=fs*0.9)

    # plotting part
    ax2.errorbar(conc, [i.n for i in dfop[s]], optode_sem[s].values, linestyle='None', marker=mark[int(s[-1])-1],
                 fillstyle='none', color=col[int(s[-1])-1], ms=6, capsize=5, label=s)
    ax2.fill_between(x=xinter, y1=interpol[0](xinter), y2=interpol[1](xinter), color=col[int(s[-1])-1], alpha=0.2, lw=0)

    # legend and axis layout / labelling
    ax2.legend(loc="upper left", bbox_to_anchor=[1, 0.9], shadow=True, fancybox=True)
    ax2.tick_params(axis='both', which='both', direction='out', labelsize=fs*0.8)

    # x,y label position
    ax2.set_xlabel('O$_2$ concentration [%air]', va='center', ha='center', fontsize=fs*0.9)
    ax2.set_ylabel('Ratio $R/G$', va='center', ha='center', rotation='vertical', fontsize=fs*0.9)

    plt.tight_layout()
    plt.show()

    return fig2, ax2


def plot_SVerrorprop(dop1_value, dop1_param, derror, f1inter_mc, RoI1_av, RoI2_av=None, dop2_value=None,
                     dop2_param=None, f2inter_mc=None, fs=11.):
    n = 1
    if RoI2_av:
        n += 1
        ls = [dop2_value, dop2_param, f2inter_mc]
        if any([i == None for i in ls]):
            raise ValueError('To plot both optodes, all data are required! Please check dop_value, dop_param, '
                             'and finter_mc')

    # -----------------------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 8), nrows=3, ncols=n, sharex=True, sharey=True, frameon=False)
    if n == 1:
        ax2[0].set_title('Optode 1', fontsize=fs*0.9)
    else:
        ax2[0][0].set_title('Optode 1', fontsize=fs*0.9), ax2[0][1].set_title('Optode 2', fontsize=fs*0.9)

    num = int(100/0.5 + 1)
    xnew = np.linspace(0, 100, num=num)

    ls_handels = list()
    if RoI1_av:
        for en, s in enumerate(dop1_value.keys()):
            name = s.split('t')[0] + 'tting ' + s.split('t')[-1]
            O2new = np.linspace(dop1_value[s]['O2 mean'].loc[0], dop1_value[s]['O2 mean'].loc[100], num=num)
            ynew = _simplifiedSV(xnew, k=dop1_param[s]['k'].mean, f=dop1_param[s]['f'].mean)
            ydata = f1inter_mc[s]

            # dashed line for bestFit
            if n == 1:
                ax2[en].plot(xnew, ynew, ls='-.', lw=1., color=col[en], label='bestFit')
                l = ax2[en].errorbar(dop1_value[s]['O2 mean'], dop1_value[s]['iratio mean'].values, capsize=6, ms=6,
                                     xerr=derror[0][s]['O2'].values, linestyle='None', marker=mark[0], color=col[en],
                                     yerr=derror[0][s]['iratio'].values, fillstyle='none', label=name)
                ax2[en].fill_between(x=O2new, y1=ydata[0](O2new), y2=ydata[1](O2new), color=col[en], alpha=0.2, lw=0)
                ls_handels.append(l)
            else:
                ax2[en][0].plot(xnew, ynew, ls='-.', lw=1., color=col[en], label='bestFit')
                l = ax2[en][0].errorbar(dop1_value[s]['O2 mean'], dop1_value[s]['iratio mean'].values, capsize=6, ms=6,
                                        xerr=derror[0][s]['O2'].values, linestyle='None', marker=mark[0], color=col[en],
                                        fillstyle='none', label=name)
                ax2[en][0].fill_between(x=O2new, y1=ydata[0](O2new), y2=ydata[1](O2new), color=col[en], lw=0, alpha=0.2)
                ls_handels.append(l)

    ls_handels = list()
    if RoI2_av:
        for en, s in enumerate(dop2_value.keys()):
            name = s.split('t')[0] + 'tting ' + s.split('t')[-1]
            O2new = np.linspace(dop2_value[s]['O2 mean'].loc[0], dop2_value[s]['O2 mean'].loc[100], num=num)
            ynew = _simplifiedSV(xnew, k=dop2_param[s]['k'].mean, f=dop2_param[s]['f'].mean)
            ydata = f2inter_mc[s]

            # dashed line for bestFit
            if n == 1:
                ax2[en].plot(xnew, ynew, ls='-.', lw=1., color=col[en], label='bestFit')
                l = ax2[en].errorbar(dop2_value[s]['O2 mean'], dop2_value[s]['iratio mean'].values, capsize=6, ms=6,
                                     xerr=derror[1][s]['O2'].values, linestyle='None', marker=mark[0], color=col[en],
                                     fillstyle='none', label=name)
                ax2[en].fill_between(x=O2new, y1=ydata[0](O2new), color=col[en], alpha=0.2, lw=0, y2=ydata[1](O2new))
                ls_handels.append(l)
            else:
                ax2[en][1].plot(xnew, ynew, ls='-.', lw=1., color=col[en], label='bestFit')
                l = ax2[en][1].errorbar(dop2_value[s]['O2 mean'], dop2_value[s]['iratio mean'].values, capsize=6, ms=6,
                                        xerr=derror[1][s]['O2'].values, linestyle='None', marker=mark[0], color=col[en],
                                        fillstyle='none', label=name)
                ax2[en][1].fill_between(x=O2new, y1=ydata[0](O2new), color=col[en], lw=0, alpha=0.2, y2=ydata[1](O2new))
                ls_handels.append(l)

    # legend and axis layout / labelling
    if n == 1:
        ax2[1].legend(handles=ls_handels, loc="upper left", bbox_to_anchor=[1, 0.9], shadow=True, fancybox=True)
    else:
        ax2[1][1].legend(handles=ls_handels, loc="upper left", bbox_to_anchor=[1, 0.9], shadow=True, fancybox=True)

    # x,y label position
    fig2.text(0.5, 0.018, 'O$_2$ concentration [%air]', va='center', ha='center', fontsize=fs*1.2)
    fig2.text(0.025, 0.55, 'Ratio $R/G$', va='center', ha='center', rotation='vertical', fontsize=fs*1.2)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)
    plt.show()

    return fig2, ax2


def plot_optode_set_SV(o, s, en, dfop, dop_para, dferr, finter_mc, fs=11):
    fig2, ax2 = plt.subplots(figsize=(5, 3), frameon=False)
    title = o + ' - ' + s
    ax2.set_title(title, loc='left', fontsize=fs * 0.9)
    xnew = np.linspace(0, 100, num=int(100 / 0.5 + 1))

    O2new = np.linspace(dfop['O2 mean'].loc[0], dfop['O2 mean'].loc[100], num=int(100 / 0.5 + 1))
    ynew = _simplifiedSV(xnew, k=dop_para['k'].mean, f=dop_para['f'].mean)

    ax2.plot(xnew, ynew, ls='-.', lw=1., color=col[en - 1], label='bestFit')
    ax2.errorbar(dfop['O2 mean'], dfop['iratio mean'].values, capsize=6, xerr=dferr['O2'].values,color=col[en - 1],
                 linestyle='None', marker=mark[0], fillstyle='none', ms=6, label=s)
    ax2.fill_between(x=O2new, y1=finter_mc[0](O2new), y2=finter_mc[1](O2new), color=col[en - 1], alpha=0.2, lw=0)

    # x,y label position
    fig2.text(0.5, 0.04, 'O$_2$ concentration [%air]', va='center', ha='center', fontsize=fs)
    fig2.text(0.025, 0.55, 'Ratio $R/G$', va='center', ha='center', rotation='vertical', fontsize=fs)
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9)
    plt.show()

    return fig2, ax2


def plot_wholeImage3D(dO2_mean, unit, pad=2):
    xx, yy = np.meshgrid(dO2_mean.index.to_numpy(), dO2_mean.columns.to_numpy())

    # 3D image of full area
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xx, yy, dO2_mean.T.fillna(limit=5, method='ffill'), cmap='magma_r', linewidth=0, vmin=0,
                           vmax=100, antialiased=False, rstride=5, cstride=10)
    cbar = fig.colorbar(surf, aspect=20, shrink=0.8)

    ax.view_init(16, 45)
    ax.tick_params(axis='x', labelsize=fs*0.9)
    ax.tick_params(axis='y', labelsize=fs*0.9)
    ax.tick_params(axis='z', labelsize=fs*0.9)
    cbar.ax.tick_params(labelsize=fs*0.8)

    ax.set_xlabel('Image height [{}]'.format(unit), fontsize=fs, labelpad=pad)
    ax.set_ylabel('Image width [{}]'.format(unit), fontsize=fs, labelpad=pad)
    ax.set_zlabel('$O_2$ concentration [%air]', fontsize=fs, labelpad=pad)

    plt.tight_layout()
    plt.draw()

    return fig, ax


def plot_optode2D(o, s, px2mm, surface, dO2_av, depth_range, width_range, figsize=(6, 2), unit='mm', fs=11, vmin=None,
                  vmax=None):
    # prepare optode for plotting; baseline correction and cropping the image to the depth and width of interest
    df_data = optodePrep2D(o=o, s=s, px2mm=px2mm, baseline=surface, dO2_av=dO2_av, depth_range=depth_range,
                           width_range=width_range)

    # resetting the axis ticks with extent
    extent = [df_data.columns[0], df_data.columns[-1],  # x-axis, e.g. columns
              df_data.index[0], df_data.index[-1]]  # y-axis, e.g. index

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    sur = ax.imshow(df_data, extent=extent, cmap='magma_r', vmin=vmin, vmax=vmax)
    if vmin is None:
        vmin = int(df_data.min().min())
    if vmax is None:
        vmax = int(df_data.max().max())
    plt.colorbar(sur, shrink=0.75, fraction=0.1, aspect=10, ticks=np.linspace(vmin, vmax, num=5))

    ax.set_xlabel('Image width [{}]'.format(unit), fontsize=fs)
    ax.set_ylabel('Image height [{}]'.format(unit), fontsize=fs)
    plt.tight_layout()

    return fig, ax


def plotLP(dO2_lp, df_ms, header_ms, depth, kshape, depth_lp, s, arg, dO2_optode=None):
    # additional information
    col_ = int(s[-1])-1

    # figure creation
    fig_lp = plt.figure(figsize=(arg['figsize']), dpi=100)
    with plt.style.context('seaborn-darkgrid'):
        ax1 = fig_lp.add_subplot(131)
        ax2 = fig_lp.add_subplot(132, sharex=ax1, sharey=ax1)
        ax3 = fig_lp.add_subplot(133, sharex=ax1, sharey=ax1)
    if dO2_optode:
        with plt.style.context('classic'):
            ax11 = fig_lp.add_axes([0.13, 0.2, 0.2, 0.2])
            ax21 = fig_lp.add_axes([0.44, 0.2, 0.2, 0.2])
            if len(dO2_lp[kshape]['square'].keys()) != 0:
                ax31 = fig_lp.add_axes([0.75, 0.2, 0.2, 0.2])

    ax1.set_title('(A) Horizontal smoothing', fontsize=fs, loc='left')
    ax2.set_title('(B) Vertical smoothing', fontsize=fs, loc='left')
    if len(dO2_lp[kshape]['square'].keys()) != 0:
        ax3.set_title('(C) Square smoothing', fontsize=fs, loc='left')

    # plot line profile
    # horizontal
    df_h = dO2_lp[kshape]['horizontal'][arg['lw']].fillna(limit=5, method='ffill').loc[depth_lp[0]: depth_lp[1]]
    ax1.plot(df_h['mean'].values, df_h.index, lw=arg['curve lw'], color=col[col_])
    ax1.fill_betweenx(df_h.index, df_h['mean'].values - df_h['SD'].values, df_h['mean'].values + df_h['SD'].values,
                      facecolor=col[col_], alpha=0.25)
    # vertical
    df_v = dO2_lp[kshape]['vertical'][arg['lw']].fillna(limit=5, method='ffill').loc[depth_lp[0]: depth_lp[1]]
    ax2.plot(df_v['mean'].values, df_v.index, lw=arg['curve lw'], color=col[col_])
    ax2.fill_betweenx(df_v.index, df_v['mean'].values - df_v['SD'].values, df_v['mean'].values + df_v['SD'].values,
                      facecolor=col[col_], alpha=0.25)
    # squared
    if len(dO2_lp[kshape]['square'].keys()) == 0:
        ax3.axis('off')
    else:
        df_s = dO2_lp[kshape]['square'][arg['lw']].fillna(limit=5, method='ffill').loc[depth_lp[0]: depth_lp[1]]
        ax3.plot(df_s['mean'].values, df_s.index, lw=arg['curve lw'], color=col[col_])
        ax3.fill_betweenx(df_s.index, df_s['mean'].values - df_s['SD'].values, df_s['mean'].values + df_s['SD'].values,
                          facecolor=col[col_], alpha=0.25)

    # ..........................................
    # 2D imshow
    if dO2_optode:
        opt_h = dO2_optode[kshape]['horizontal']
        extent = [opt_h.columns[0], opt_h.columns[-1],  # x-axis, e.g. columns
                  opt_h.index[-1], opt_h.index[0]]  # y-axis, e.g. index
        op1 = ax11.imshow(opt_h, extent=extent, aspect=arg['aspect'], cmap=arg['cmap'], vmin=arg['vmin op'],
                          vmax=arg['vmax op'])
        op2 = ax21.imshow(dO2_optode[kshape]['vertical'], extent=extent, aspect=arg['aspect'], cmap=arg['cmap'],
                          vmin=arg['vmin op'], vmax=arg['vmax op'])
        if len(dO2_lp[kshape]['square'].keys()) != 0:
            op3 = ax31.imshow(dO2_optode[kshape]['square'], extent=extent, aspect=arg['aspect'], cmap=arg['cmap'],
                              vmin=arg['vmin op'], vmax=arg['vmax op'])

        # color bar
        fig_lp.colorbar(op1, aspect=10, shrink=0.8, ax=ax11)
        fig_lp.colorbar(op2, aspect=10, shrink=0.8, ax=ax21)
        if len(dO2_lp[kshape]['square'].keys()) != 0:
            fig_lp.colorbar(op3, aspect=10, shrink=0.8, ax=ax31)

    # ..........................................
    # microsensor
    ax1.plot(df_ms[header_ms[1]].to_numpy(), depth, lw=arg['curve lw'], color='black', label='microsensor')
    ax2.plot(df_ms[header_ms[1]].to_numpy(), depth, lw=arg['curve lw'], color='black', label='microsensor')
    if len(dO2_lp[kshape]['square'].keys()) != 0:
        ax3.plot(df_ms[header_ms[1]].to_numpy(), depth, lw=arg['curve lw'], color='black', label='microsensor')

    # ..........................................
    # adjust axes
    ax1.set_xlim(arg['vmin'], arg['vmax'])
    ax1.set_ylim(df_h.index[-1] * 1.05, df_h.index[0] * 1.05)
    ax1.tick_params(labelsize=arg['fontsize']*0.9)
    ax2.tick_params(labelsize=arg['fontsize']*0.9)
    if len(dO2_lp[kshape]['square'].keys()) != 0:
        ax3.tick_params(labelsize=arg['fontsize']*0.9)
    if dO2_optode:
        ax11.tick_params(labelsize=arg['fontsize']*0.7)
        ax21.tick_params(labelsize=arg['fontsize']*0.7)
        if len(dO2_lp[kshape]['square'].keys()) != 0:
            ax31.tick_params(labelsize=arg['fontsize']*0.7)

        ax11.set_xlabel('Width [mm]', fontsize=arg['fontsize']*0.7)
        ax11.set_ylabel('Height [mm]', fontsize=arg['fontsize']*0.7)
        ax21.set_xlabel('Width [mm]', fontsize=arg['fontsize']*0.7)
        ax21.set_ylabel('Height [mm]', fontsize=arg['fontsize']*0.7)
        if len(dO2_lp[kshape]['square'].keys()) != 0:
            ax31.set_xlabel('Width [mm]', fontsize=arg['fontsize']*0.7)
            ax31.set_ylabel('Height [mm]', fontsize=arg['fontsize']*0.7)
    fig_lp.text(0.4, 0.02, '$O_2$ concentration [%air]', fontsize=arg['fontsize'])
    fig_lp.text(0.01, 0.48, 'Depth [mm]', fontsize=arg['fontsize'], rotation='vertical')

    fig_lp.subplots_adjust(bottom=0.12, right=0.95, top=0.95, left=0.05, wspace=0.2, hspace=0.2)

    return fig_lp


def plot_penetrationDepth(depth, ls_kernel, arg):
    if isinstance(ls_kernel[0], tuple):
        kernel_s = [k[1] for k in ls_kernel]
    else:
        kernel_s = ls_kernel
    # .....................
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for en, c in enumerate(depth.columns):
        ax.plot(kernel_s, depth[c], lw=1., ls='-.', marker=arg['marker'][en], ms=7,
                color=arg['colors'][en], fillstyle='none', label=c.split('-')[0] + ' blur')

    ax.legend(loc=0, frameon=True, fancybox=True, fontsize=fs * 0.8)
    ax.tick_params(axis='both', labelsize=fs * 0.8)
    ax.set_xlabel('kernel size', fontsize=fs)
    ax.set_ylabel('$O_2$ penetration depth [mm]', fontsize=fs)
    plt.tight_layout()

    return fig


# =====================================================================================
def crop_optode(dratio, RoI1, RoI2):
    # optode 1
    if RoI1 == None:
        optode1 = None
    else:
        optode1 = dict()
        for en, c in enumerate(dratio.keys()):
            ls_av = list()
            for av in range(len(RoI1)):
                height = RoI1[av][1][1] - RoI1[av][0][1]
                im_ratio = dratio[c][0][RoI1[av][0][1]:RoI1[av][1][1] + 1]
                ls_av.append(np.stack([im_ratio[n][RoI1[av][0][0]:RoI1[av][2][1] + 1] for n in np.arange(height + 1)],
                                      axis=0))
            optode1[c] = ls_av

    # -------------------------------------------------------------------------
    # optode 2
    if RoI2 == None:
        optode2 = None
    else:
        optode2 = dict()
        for en, c in enumerate(dratio.keys()):
            ls_av = list()
            for av in range(len(RoI2)):
                height2 = RoI2[av][1][1] - RoI2[av][0][1]
                im_ratio2 = dratio[c][1][RoI2[av][0][1]:RoI2[av][1][1] + 1]
                ls_av.append(np.stack([im_ratio2[n][RoI2[av][0][0]:RoI2[av][2][1] + 1] for n in np.arange(height2 + 1)],
                                      axis=0))
            optode2[c] = ls_av

    return optode1, optode2


def image_resolution(px, dist_mm, inch=None):
    px2mm = px / dist_mm * 1
    if inch:
        dpi = px / inch
    else:
        dpi = None

    return px2mm, dpi


def px2mm_conversion(df, px2mm, surface):
    ind_new = df.index.to_numpy() / px2mm - surface
    col_new = df.columns.to_numpy() / px2mm
    df.index, df.columns = ind_new, col_new
    return df


def round_decimals_up(number, decimals=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def round_decimals_down(number, decimals=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


# =====================================================================================
def averaging_areas(doptode_set):
    l = dict(map(lambda c: (c, pd.DataFrame([(np.mean(doptode_set[c][av]), np.std(doptode_set[c][av]))
                                             for av in range(len(doptode_set[c]))], columns=['mean', 'SD_area'])),
                 doptode_set.keys()))
    dfop_set = pd.concat(l, axis=0).sort_index(axis=0)

    return dfop_set


def averaging_deriv(ls_x):
    der = 1/len(ls_x) * ls_x
    return der


def optode_normalization(dfop):
    optode_norm = dict(map(lambda s: (s, dfop[s].loc[0] / dfop[s]), dfop.keys()))
    # update calibration point zero
    for s in optode_norm.keys():
        sd = ((dfop[s][0].std_dev / dfop[s][0].n)**2)*2
        optode_norm[s].loc[0] = ufloat(dfop[s][0].n / dfop[s][0].n, np.sqrt(sd) * dfop[s][0].n / dfop[s][0].n)
    return optode_norm


def interpolation_SD(conc, dfop, s, method='cubic'):
    f_min = interp1d(conc, [i.n-i.s for i in dfop[s]], kind=method)
    f_max = interp1d(conc, [i.n+i.s for i in dfop[s]], kind=method)
    return f_min, f_max


def interpolation_SDmc(df, s, method='cubic'):
    I_min = interp1d(df[s]['O2 mean'].values, (df[s]['iratio mean'] - df[s]['iratio SD']).values, kind=method)
    I_max = interp1d(df[s]['O2 mean'].values, (df[s]['iratio mean'] + df[s]['iratio SD']).values, kind=method)
    return I_min, I_max


def channel_division(dconc, dch_num, dch_denom, s):
    dratio = dict()
    for c in dconc[s]:
        dratio[c] = [dch_num[s][str(c)+'%'][n] / dch_denom[s][str(c)+'%'][n] for n in range(len(dch_num[s][str(c)+'%']))]
    return dratio


def ratiometric_intensity(path, crop_op, channel, RoI1=None, RoI2=None):
    # RoI are areas defined anti-clockwise starting from the top left corner with P(col / ind)
    if crop_op:
        pass
    else:
        RoI_op_ = RoI1 + RoI2
        crop_op = [[(RoI_op_[o][p][0] + 5, RoI_op_[o][p][1] + 5) for p in range(len(RoI_op_[o]))]
                   for o in range(len(RoI_op_))]

    # -------------------------------
    # RoI for different settings
    height = list(map(lambda n: (crop_op[n][1][1] - crop_op[n][0][1]), range(len(crop_op))))

    # load all calibration images - collect information
    dict_red, dict_green, dict_conc = load_calibration_info(path=path, RoI1=crop_op, height=height, channel=channel,
                                                            server=False)

    # calculating the ratio R/G of the whole optode
    dratio = dict(map(lambda k: (k, channel_division(dconc=dict_conc, dch_num=dict_red, dch_denom=dict_green, s=k)),
                      dict_red.keys()))

    # combine cropped info
    optode1 = dict(map(lambda s: (s, crop_optode(dratio[s], RoI1=RoI1, RoI2=RoI2)[0]), dratio.keys()))
    optode2 = dict(map(lambda s: (s, crop_optode(dratio[s], RoI1=RoI1, RoI2=RoI2)[1]), dratio.keys()))

    # determine number of pixels within the defined RoI = sample size
    if RoI1:
        npx1 = (RoI1[0][1][1] - RoI1[0][0][1]) * (RoI1[0][2][0] - RoI1[0][0][0])
    else:
        npx1 = 0
    if RoI2:
        npx2 = (RoI2[0][1][1] - RoI2[0][0][1]) * (RoI2[0][2][0] - RoI2[0][0][0])
    else:
        npx2 = 0

    # ----------------------------------------------------------
    # signal averaged within RoI used as start/input signal for uncertainty propagation averaging each RoI for all
    # optodes and settings
    if RoI1:  # optode 1
        dfop1_set1 = averaging_areas(doptode_set=optode1['set1'])
        dfop1_set2 = averaging_areas(doptode_set=optode1['set2'])
        dfop1_set3 = averaging_areas(doptode_set=optode1['set3'])

        conc = dfop1_set1.index.levels[0].to_numpy()
        dfop1 = dict({'set1': [ufloat(dfop1_set1.loc[i, 'mean'], dfop1_set1.loc[i, 'SD_area'])
                               for i in dfop1_set1.index],
                      'set2': [ufloat(dfop1_set2.loc[i, 'mean'], dfop1_set2.loc[i, 'SD_area'])
                               for i in dfop1_set2.index],
                      'set3': [ufloat(dfop1_set3.loc[i, 'mean'], dfop1_set3.loc[i, 'SD_area'])
                               for i in dfop1_set3.index]})
        dfop1 = pd.DataFrame(dfop1, index=conc)
    else:
        dfop1 = None
    if RoI2:  # optode 2
        dfop2_set1 = averaging_areas(doptode_set=optode2['set1'])
        dfop2_set2 = averaging_areas(doptode_set=optode2['set2'])
        dfop2_set3 = averaging_areas(doptode_set=optode2['set3'])

        conc = dfop2_set1.index.levels[0].to_numpy()
        dfop2 = dict({'set1': [ufloat(dfop2_set1.loc[i, 'mean'], dfop2_set1.loc[i, 'SD_area'])
                               for i in dfop2_set1.index],
                      'set2': [ufloat(dfop2_set2.loc[i, 'mean'], dfop2_set2.loc[i, 'SD_area'])
                               for i in dfop2_set2.index],
                      'set3': [ufloat(dfop2_set3.loc[i, 'mean'], dfop2_set3.loc[i, 'SD_area'])
                               for i in dfop2_set3.index]})
        dfop2 = pd.DataFrame(dfop2, index=conc)
    else:
        dfop2 = None

    # prepare for output
    dfoptode = [dfop1, dfop2]
    para = dict({'sample size': (npx1, npx2), 'concentration': conc, 'ch1': dict_red, 'ch2': dict_green})
    return dfoptode, para


def reduce_dict(name, dint1, dint2=None, nopt=1, option='ratio'):
    if option == 'ratio':
        dop1 = dict(map(lambda s: (s, np.divide(dint1[s][name][0], dint2[s][name][0])), dint1.keys()))
        if nopt > 1:
            dop2 = dict(map(lambda s: (s, np.divide(dint1[s][name][1], dint2[s][name][1])), dint1.keys()))
        else:
            dop2 = None
    else:
        dop1 = dict(map(lambda s: (s, dint1[s][name][0]), dint1.keys()))
        if nopt > 1:
            dop2 = dict(map(lambda s: (s, dint1[s][name][1]), dint1.keys()))
        else:
            dop2 = None

    dint = dict({'optode1': dop1, 'optode2': dop2})
    return dint


def splitImage(path, RoI_op):
    # RoI for different settings
    height = dict(map(lambda o: (o, RoI_op[o][1][1] - RoI_op[o][0][1]), range(len(RoI_op))))
    dict_red, dict_green = load_files(path, RoI_op, height)

    # split into smaller dictionaries
    name = list(dict_red['set1'].keys())[0]
    dint_red = reduce_dict(name=name, dint1=dict_red, dint2=None, nopt=len(RoI_op), option='single')
    dint_green = reduce_dict(name=name, dint1=dict_green, dint2=None, nopt=len(RoI_op), option='single')
    dint_ratio = reduce_dict(name=name, dint1=dict_red, dint2=dict_green, nopt=len(RoI_op), option='ratio')

    return dint_red, dint_green, dint_ratio


def split2statics(dO2):
    # mean value
    dic_av = dict(map(lambda o:
                      (o, dict(map(lambda s:
                                   (s, pd.DataFrame(list(map(lambda j: [i.n if i is not np.nan else i
                                                                        for i in dO2[o][s][j]], dO2[o][s].columns)),
                                                    columns=dO2[o][s].index, index=dO2[o][s].columns).T),
                                   dO2[o].keys()))), dO2.keys()))
    # standard error
    dic_sd = dict(map(lambda o:
                      (o, dict(map(lambda s:
                                   (s, pd.DataFrame(list(map(lambda j: [i.s if i is not np.nan else i
                                                                        for i in dO2[o][s][j]], dO2[o][s].columns)),
                                                    columns=dO2[o][s].index, index=dO2[o][s].columns).T),
                                   dO2[o].keys()))), dO2.keys()))

    return dic_av, dic_sd


def line_profile_v1(df, lp, lw):
    if df.empty is True:
        df_lp = None
    else:
        # find closest value in df.columns
        diff_min, diff_max = (lp - lw / 2) - df.columns, (lp + lw / 2) - df.columns

        for en, i in enumerate(diff_min):
            if i == min(np.abs(diff_min)):
                pos_min = (en, df.columns[en])
        for en, i in enumerate(diff_max):
            if i == min(np.abs(diff_max)):
                pos_max = (en, df.columns[en])

        if pos_min:
            pass
        else:
            pos_min = (None, None)
        if pos_max:
            pass
        else:
            pos_max = (None, None)

        if pos_min == pos_max:
            df_lp = pd.DataFrame(df[pos_min[1]])
        else:
            df_lp = df.loc[:, pos_min[1]:pos_max[1]]
    return df_lp


def optodePrep2D(o, s, dO2_av, px2mm, baseline, depth_range=None, width_range=None):
    # image preparation and cropping image depth/width
    if isinstance(dO2_av, dict):
        df_ex = dO2_av[o][s]
    else:
        df_ex = dO2_av

    xnew = df_ex.index
    df = df_ex.copy()
    df.index = reversed(xnew)

    if depth_range is None:
        df_ = df
    else:
        px_range = np.arange(0, len(df.index) + 1, step=1)
        px_range_mm = px_range / px2mm - baseline[int(o.split('e')[-1])-1]

        crop_px1 = list()
        for en, p in enumerate(px_range_mm):

            if p.round(1) == depth_range[0]:

                crop_px1.append(en)
        crop_px2 = list()
        for en, p in enumerate(px_range_mm):
            if p.round(1) == depth_range[1]:
                crop_px2.append(en)

        crop_px = int(np.mean(crop_px1)), int(np.mean(crop_px2))

        df_ = df.loc[df.index[min(crop_px)]:df.index[max(crop_px)], :]
        df_.index = reversed(df_ex.index[min(crop_px):max(crop_px) + 1])

    if width_range is None:
        df_data = df_
    else:
        df_data = df_.loc[:, width_range[0]:width_range[1]]

    return df_data


def sem_optode(dfop, RoI, conc):
    n = np.sqrt(sum([(RoI[i][1][1] - RoI[i][0][1])*(RoI[i][2][0] - RoI[i][0][0]) for i in range(len(RoI))]))
    dfop_sem = dict(map(lambda s: (s, [i.s/n for i in dfop[s]]), dfop.keys()))

    optode_sem = pd.concat(list(map(lambda s: pd.DataFrame([np.mean(dfop_sem[s][c:(c+1)]) for c in range(len(conc))],
                                                           index=conc, columns=[s]), dfop.keys())), axis=1)
    return optode_sem, n


def uncertainty(para, RoI1, RoI2, conc, dfop1=None, dfop2=None, method='cubic'):
    # interpolation for SD
    if isinstance(dfop1, pd.DataFrame):
        f_op1 = dict(map(lambda s: (s, interpolation_SD(conc=para['concentration'], dfop=dfop1, s=s, method=method)),
                         dfop1.columns))
        # standard error of the mean
        optode1_sem, n1 = sem_optode(dfop=dfop1, RoI=RoI1, conc=conc)
        # combine for output
        uncer_op1 = dict({'SD_interpol': f_op1, 'sem': optode1_sem, 'sample size': n1})
    else:
        uncer_op1 = None

    if isinstance(dfop2, pd.DataFrame):
        f_op2 = dict(map(lambda s: (s, interpolation_SD(conc=para['concentration'], dfop=dfop2, s=s, method=method)),
                         dfop2.columns))
        # standard error of the mean
        optode2_sem, n2 = sem_optode(dfop=dfop2, RoI=RoI2, conc=conc)
        # combine for output
        uncer_op2 = dict({'SD_interpol': f_op2, 'sem': optode2_sem, 'sample size': n2})
    else:
        uncer_op2 = None

    return uncer_op1, uncer_op2


def lin_propagation(dfop1, dfop2, n1, n2, RoI1, RoI2, conc):
    # normalization
    if RoI1:  # optode 1
        optode1_norm = optode_normalization(dfop=dfop1)
    else:
        optode1_norm = None

    if RoI2:  # optode 2
        optode2_norm = optode_normalization(dfop=dfop2)
    else:
        optode2_norm = None
    optode_norm = [optode1_norm, optode2_norm]

    # standard error of the mean
    if RoI1:
        optode1_norm_SEM = dict(map(lambda s: (s, pd.DataFrame([i.s / n1 for i in optode1_norm[s]],
                                                               index=optode1_norm[s].index)), optode1_norm.keys()))
        # interpolation for SD
        fnorm_op1 = dict(map(lambda s: (s, interpolation_SD(conc=conc, dfop=optode1_norm, s=s)), optode1_norm.keys()))
    else:
        optode1_norm_SEM, fnorm_op1 = None, None

    if RoI2:
        optode2_norm_SEM = dict(map(lambda s: (s, pd.DataFrame([i.s / n2 for i in optode2_norm[s]],
                                                               index=optode2_norm[s].index)), optode2_norm.keys()))
        # interpolation for SD
        fnorm_op2 = dict(map(lambda s: (s, interpolation_SD(conc=conc, dfop=optode2_norm, s=s)), optode2_norm.keys()))
    else:
        optode2_norm_SEM, fnorm_op2 = None, None

    return optode_norm, optode1_norm_SEM, optode2_norm_SEM, fnorm_op1, fnorm_op2


def mc_propagation(conc, dfop, optode_norm, optode_norm_SEM, RoI, uncer_op):
    dic_optode_value = dict()
    dic_optode_param = dict()
    for s in dfop.columns:
        if RoI:
            [dic_optode_param[s], dic_optode_value[s]] = mcerp_simplifiedSVFit(optode=optode_norm[s].to_numpy(),
                                                                               conc=conc)

    for s in dic_optode_param.keys():
        if RoI:
            dic_optode_param[s]['I0'] = dfop.loc[0][s]

    # -------------------------------------------------------------------
    # uncertainty propagation for SEM
    # intensity
    iratio_normSEM = dict(map(lambda s: (s, pd.Series([ufloat(optode_norm[s].loc[c].n, optode_norm_SEM[s].loc[c][0])
                                                       for c in optode_norm[s].index], index=optode_norm[s].index,
                                                      name=s)), optode_norm.keys()))

    # concentration
    ox_normSEM = dict(map(lambda s: (s, dic_optode_value[s]['O2 SD'] / uncer_op['sample size']),
                          dic_optode_value.keys()))

    optode_normSEM = dict(map(lambda s: (s, pd.concat([ox_normSEM[s], iratio_normSEM[s]], axis=1,
                                                      keys=['O2 SEM', 'iratio SEM'])), iratio_normSEM.keys()))

    return dic_optode_value, dic_optode_param, optode_normSEM


# =====================================================================================
def _simplifiedSV(x, f, k):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param k:   np.float
    :param f:   np.float
    :return: iratio:    normalized signal i0/i
    """
    return 1 / (f / (1. + k*x) + (1.-f))


def _simplified_SVFit_1run(data, conc, par0=None):
    simply_sv = Model(_simplifiedSV)
    if par0:
        params_sens = simply_sv.make_params(k=par0['k'], f=par0['f'])
    else:
        params_sens = simply_sv.make_params(k=0.165, f=0.887)

    params_sens['k'].min = 0.
    params_sens['f'].max = 1.
    params_sens['f'].vary = True
    params_sens['k'].vary = True

    # use i0/i data for fit and re-calculate i afterwards
    # full concentration range
    result = simply_sv.fit(data, params_sens, x=conc, nan_policy='omit')

    return result


def mcerp_simplifiedSVFit(optode, conc):
    # use simplifiedSV_run1 to calculate best fit
    res = _simplified_SVFit_1run(data=[i.n for i in optode], conc=conc)

    # evaluate the covariance matrix of your parameters
    covariance = res.covar

    # draw random samples from a normal multivariate distribution using the best value of your parameters
    # and their covariance matrix
    f = N(res.params['f'].value, res.params['f'].stderr**2) # stderr**2 := cov(f,f)
    k = N(res.params['k'].value, res.params['k'].stderr**2)
    y = [N(o.n, o.s) for o in optode]

    params = dict({'f': f, 'k': k, 'covariance': covariance, 'fit result': res})

    # calculate x for each point of the sample
    O2_calc = O2_analysis_v2(f=f, k=k, iratio=y)

    # estimate the mean and standard deviation of x
    ox_out = [(O2_calc[ox].mean, np.sqrt(O2_calc[ox].var), optode[ox].n, optode[ox].s) for ox in range(len(conc))]
    out = pd.DataFrame(ox_out, index=conc, columns=['O2 mean', 'O2 SD', 'iratio mean', 'iratio SD'])

    return params, out


def o2_calculation(inp, dict_ratio_run1, dict_ratio_run2, dpara, surface, px2mm, splitdata=True, run=2, vmin=-50,
                   vmax=150):
    o, s, run = inp.split(',')[0].strip(), inp.split(',')[1].strip(), int(run)
    if run == 1:
        dratio = dict_ratio_run1
    else:
        dratio = dict_ratio_run2

    dO2_calc = dict()
    for o in dratio.keys():
        if dratio[o]:
            dic_cal = dict(map(lambda s: (s, O2_analysis_area(para=dpara[o][s], iratio=dratio[o][s])),
                               dratio[o].keys()))
            dO2_calc[o] = dic_cal

    # post-processing
    dO2, dO2_av, dO2_SD = postprocessing_v1(dO2_calc=dO2_calc, px2mm=px2mm, surface=surface, split=splitdata, vmin=vmin,
                                            vmax=vmax)

    return dO2, dO2_av, dO2_SD


def O2_analysis_v2(f, k, iratio):
    """
    :param f:   mcerp.UncertainVariable contaning a normal distributed sample of values around the best value of the
                fit parameter f and its covariance value as sigma
    :param k:   mcerp.UncertainVariable contaning a normal distributed sample of values around the best value of
                the fit parameter k and its covariance value as sigma
    :param iratio: list of mcerp.UncertainVariables containing a normal distributed sample of the intensity ratio
                (mu is the average value and sigma is the proagated error)
    return x:
    """

    # mean O2 concentration
    x = [1/k * (f / ((1/y) + f -1) -1) for y in iratio]
    return x


def O2_analysis_area(para, iratio, iratio_std=None, int_type='norm'):
    """
    :param f:   mcerp.UncertainVariable contaning a normal distributed sample of values around the best value of the
                fit parameter f and its covariance value as sigma
    :param k:   mcerp.UncertainVariable contaning a normal distributed sample of values around the best value of the
                fit parameter k and its covariance value as sigma
    :param iratio: array of mcerp.UncertainVariables containing a normal distributed sample of the intensity ratio
                (mu is the average value and sigma is the proagated error) or only mean values as np.float64
    return x:
    """
    # create ufloat for uncertainty propagation via parameter
    f_mp = ufloat(para.loc['f'][0], para.loc['f'][1])
    k_mp = ufloat(para.loc['k'][0], para.loc['k'][1])

    if iratio_std is None:
        iratio_std = np.array(np.empty(iratio.size)).reshape(iratio.shape)
        iratio_std[:] = np.NaN

    if int_type == 'norm':
        if isinstance(iratio, np.ndarray) and isinstance(iratio_std, np.ndarray):
            int_arr = unumpy.uarray(iratio, iratio_std)
        elif isinstance(iratio, np.ndarray) is False and isinstance(iratio_std, np.ndarray):
            int_arr = unumpy.uarray(iratio.to_numpy(), iratio_std)
        elif isinstance(iratio, np.ndarray) and isinstance(iratio_std, np.ndarray) is False:
            int_arr = unumpy.uarray(iratio, iratio_std.to_numpy())
        else:
            int_arr = unumpy.uarray(np.array(iratio.to_numpy()), np.array(iratio_std.to_numpy()))
    else:
        i0_mp = ufloat(para.loc['I0'][0], para.loc['I0'][1])
        if isinstance(iratio, (np.ndarray, np.generic)):
            iratio_arr = unumpy.uarray(iratio, np.array(np.zeros(shape=(iratio.shape))))
        else:
            iratio_arr = unumpy.uarray(iratio.values, np.array(np.zeros(shape=(iratio.shape))))
        int_arr = iratio_arr / i0_mp

    # intermediate value calculation for x = 1/k * (np.divide(f, np.divide(1, inorm) + f - 1) - 1)
    a = int_arr + f_mp - 1
    b = f_mp / a - 1

    # final O2 concentration
    x = 1 / k_mp * b
    df_x = pd.DataFrame(x, index=pd.DataFrame(iratio).index, columns=pd.DataFrame(iratio).columns)

    return df_x


# =====================================================================================
def fsigmoid(x, a, b, c):
    return c / (1.0 + np.exp(-a * (x - b)))


def interpolation_microsensor(df_ms, profile_ex):
    smodel = Model(fsigmoid)

    # interpolation of microsensor to step width of optode
    params = smodel.make_params(a=-15, b=1, c=50)
    res_ms = smodel.fit(df_ms.loc[1:16, :]['Intensity'].to_numpy(), x=df_ms.loc[1:16, :]['Depth (mm)'].to_numpy(),
                        params=params)

    xnew = profile_ex.index
    ydata = fsigmoid(x=xnew, a=res_ms.best_values['a'], b=res_ms.best_values['b'], c=res_ms.best_values['c'])
    data_ms = pd.DataFrame(ydata, index=xnew)
    data_ms.columns = ['microsensor']

    return data_ms


def geometric_intersection(treshold, dd, column):
    # generate curve
    second_line = LineString(np.column_stack((dd.index, [treshold]*dd.shape[0])))
    first_line = LineString(np.column_stack((dd.index, dd[column].to_numpy())))

    # geometric determination of intersection points
    intersection = first_line.intersection(second_line)
    try:
        xdata = LineString(intersection).xy
    except:
        xdata = intersection.xy

    return xdata


def penetration_depth(dO2_lp, ls_kernel, df_ms, treshold):
    # combine relevant line profiles
    dprofile = dict()
    for kshape in ls_kernel:
        if len(dO2_lp[kshape]['square'].keys()) != 0:
            depth = pd.concat([dO2_lp[kshape]['vertical'][0], dO2_lp[kshape]['horizontal'][0],
                               dO2_lp[kshape]['square'][0]], axis=1)
            col = dO2_lp[kshape].keys()
        else:
            depth = pd.concat([dO2_lp[kshape]['vertical'][0], dO2_lp[kshape]['horizontal'][0]], axis=1)
            col = ['vertical', 'horizontal']
        depth.columns = [i + '-' + j for i in col for j in ['mean', 'SD']]
        dprofile[kshape] = depth

    # exponential decay for interpolation of micro-sensor data close to the transition
    data_ms = interpolation_microsensor(df_ms=df_ms, profile_ex=dprofile[ls_kernel[0]])

    # geometric intersection of line profile and O2 threshold for penetration depth
    dd = dict(map(lambda k: (k, pd.concat([dprofile[k].filter(like='mean'), data_ms], axis=1)), ls_kernel))

    # minimal line profile
    dd_min = dict(map(lambda k: (k, pd.concat([pd.DataFrame([dprofile[k][c + '-mean'] - dprofile[k][c + '-SD']
                                                             for c in col], index=col).T, data_ms['microsensor']],
                                              axis=1)), ls_kernel))

    # maximal line profile
    dd_max = dict(map(lambda k: (k, pd.concat([pd.DataFrame([dprofile[k][c + '-mean'] + dprofile[k][c + '-SD']
                                                             for c in col], index=col).T, data_ms['microsensor']],
                                              axis=1)), ls_kernel))

    ydepth = pd.concat([pd.DataFrame([geometric_intersection(treshold=treshold, dd=dd[k], column=d)[0][0]
                                      for d in dd[k].columns], index=dd[k].columns) for k in ls_kernel], axis=1).T
    ydepth_min = pd.concat([pd.DataFrame([geometric_intersection(treshold=treshold, dd=dd_min[k], column=d)[0][0]
                                          for d in dd_min[k].columns], index=dd_min[k].columns) for k in ls_kernel],
                           axis=1).T
    ydepth_max = pd.concat([pd.DataFrame([geometric_intersection(treshold=treshold, dd=dd_max[k], column=d)[0][0]
                                          for d in dd_max[k].columns], index=dd_max[k].columns) for k in ls_kernel],
                           axis=1).T

    ydepth.index, ydepth_min.index, ydepth_max.index = ls_kernel, ls_kernel, ls_kernel
    ydepth.columns = [i.split('-')[0] for i in ydepth.columns]

    return ydepth, ydepth_min, ydepth_max


# =====================================================================================
def saving_res(save_name, conc, crop_op, RoI1_av, RoI2_av, df_initial, df_norm, dop1_param, dop2_param, dop1_value,
               dop2_value, op1_normSEM, op2_normSEM):

    # open h5 file
    f = h5py.File(save_name, "w")

    # -----------------------------
    # [group creation]
    # header
    grp_header = f.create_group('header')
    supgrp_nRoI = grp_header.create_group("Pixels for optode")
    supgrp_nRoI1 = supgrp_nRoI.create_group("optode1")
    supgrp_nRoI2 = supgrp_nRoI.create_group("optode2")
    supgrp_RoI = grp_header.create_group("RoI for optode")
    supgrp_RoI1 = supgrp_RoI.create_group("optode1")
    supgrp_RoI2 = supgrp_RoI.create_group("optode2")
    supgrp_conc = grp_header.create_group("concentration point")

    # data group
    grp_data = f.create_group("data")

    supgrp_av = grp_data.create_group('averaged')
    supgrp_av1 = supgrp_av.create_group('optode1')
    supgrp_av2 = supgrp_av.create_group('optode2')

    supgrp_norm = grp_data.create_group('normalized')
    supgrp_norm1 = supgrp_norm.create_group('optode1')
    supgrp_norm2 = supgrp_norm.create_group('optode2')

    # group related to fit process
    grp_fit = f.create_group("fit")
    supgrp_params = grp_fit.create_group("parameter")
    supgrp_params1 = supgrp_params.create_group('optode1')
    supgrp_params2 = supgrp_params.create_group('optode2')

    supgrp_cov = grp_fit.create_group("covariance matrix")
    supgrp_cov1 = supgrp_cov.create_group('optode1')
    supgrp_cov2 = supgrp_cov.create_group('optode2')

    supgrp_chi = grp_fit.create_group("reduced chi-square")
    supgrp_chi1 = supgrp_chi.create_group('optode1')
    supgrp_chi2 = supgrp_chi.create_group('optode2')

    supgrp_values = grp_fit.create_group("values")
    supgrp_values1 = supgrp_values.create_group('optode1')
    supgrp_v1_o2av = supgrp_values1.create_group('O2 mean')
    supgrp_v1_o2sd = supgrp_values1.create_group('O2 SD')
    supgrp_v1_o2sem = supgrp_values1.create_group('O2 SEM')
    supgrp_v1_iav = supgrp_values1.create_group('iratio mean')
    supgrp_v1_isd = supgrp_values1.create_group('iratio SD')
    supgrp_v1_isem = supgrp_values1.create_group('iratio SEM')
    supgrp_values2 = supgrp_values.create_group('optode2')
    supgrp_v2_o2av = supgrp_values2.create_group('O2 mean')
    supgrp_v2_o2sd = supgrp_values2.create_group('O2 SD')
    supgrp_v2_o2sem = supgrp_values2.create_group('O2 SEM')
    supgrp_v2_iav = supgrp_values2.create_group('iratio mean')
    supgrp_v2_isd = supgrp_values2.create_group('iratio SD')
    supgrp_v2_isem = supgrp_values2.create_group('iratio SEM')
    # --------------------------------------------------------
    # [fill groups]
    # --------------------------------------------------------
    # header
    # Pixels for optode
    supgrp_nRoI1.create_dataset('RoI1', data=np.array(crop_op[0]))
    supgrp_nRoI2.create_dataset('RoI2', data=np.array(crop_op[1]))
    # concentration
    supgrp_conc.create_dataset('concentration', data=conc)
    # RoI within optode
    supgrp_RoI1.create_dataset('RoI1', data=np.array(RoI1_av))
    supgrp_RoI2.create_dataset('RoI1', data=np.array(RoI2_av))

    # ------------------------------
    # data
    # supgroup - averaged data
    for s in df_initial[0].columns:
        v = np.array([[i.n for i in df_initial[0][s].values], [i.s for i in df_initial[0][s].values]])
        supgrp_av1.create_dataset(str(s), data=np.array(v))
    for s in df_initial[1].columns:
        v = np.array([[i.n for i in df_initial[1][s].values], [i.s for i in df_initial[1][s].values]])
        supgrp_av2.create_dataset(str(s), data=np.array(v))

    # ------------------------------
    # supgroup - normalized data
    for s in df_norm[0].keys():
        v = [[i.n for i in df_norm[0][s].values], [i.s for i in df_norm[0][s].values]]
        supgrp_norm1.create_dataset(str(s), data=np.array(v))
    for s in df_norm[1].keys():
        v = [[i.n for i in df_norm[1][s].values], [i.s for i in df_norm[1][s].values]]
        supgrp_norm2.create_dataset(str(s), data=np.array(v))

    # ------------------------------
    # supgroup - fit parameters
    for s in dop1_param.keys():
        v = [(dop1_param[s][l].mean, dop1_param[s][l].std) for l in ['f', 'k']]
        v += [(dop1_param[s]['I0'].n, dop1_param[s]['I0'].s)]
        supgrp_params1.create_dataset(str(s), data=np.array(v))
    for s in dop2_param.keys():
        v = [(dop2_param[s][l].mean, dop2_param[s][l].std) for l in ['f', 'k']]
        v += [(dop2_param[s]['I0'].n, dop2_param[s]['I0'].s)]
        supgrp_params2.create_dataset(str(s), data=np.array(v))

    # ------------------------------
    # supgroup - covariance matrix
    for s in dop1_param.keys():
        supgrp_cov1.create_dataset(str(s), data=np.array(dop1_param[s]['covariance']))
    for s in dop2_param.keys():
        supgrp_cov2.create_dataset(str(s), data=np.array(dop2_param[s]['covariance']))

    # ------------------------------
    # supgroup - reduces chi-square
    for s in dop1_param.keys():
        supgrp_chi1.create_dataset(str(s), data=np.array(dop1_param[s]['fit result'].redchi))
    for s in dop1_param.keys():
        supgrp_chi2.create_dataset(str(s), data=np.array(dop2_param[s]['fit result'].redchi))

    # ------------------------------
    # supgroup - fit values
    # columns - [O2 mean, O2 SD, iratio mean, iratio SD, O2 SEM, iratio SEM]
    for s in dop1_value.keys():
        supgrp_v1_o2av.create_dataset(str(s), data=dop1_value[s]['O2 mean'].to_numpy())
        supgrp_v1_o2sd.create_dataset(str(s), data=dop1_value[s]['O2 SD'].to_numpy())
        supgrp_v1_o2sem.create_dataset(str(s), data=op1_normSEM[s]['O2 SEM'].to_numpy())

        supgrp_v1_iav.create_dataset(str(s), data=dop1_value[s]['iratio mean'].to_numpy())
        supgrp_v1_isd.create_dataset(str(s), data=dop1_value[s]['iratio SD'].to_numpy())
        supgrp_v1_isem.create_dataset(str(s), data=[i.s for i in op1_normSEM['set1']['iratio SEM']])

    for s in dop2_value.keys():
        supgrp_v2_o2av.create_dataset(str(s), data=dop2_value[s]['O2 mean'].to_numpy())
        supgrp_v2_o2sd.create_dataset(str(s), data=dop2_value[s]['O2 SD'].to_numpy())
        supgrp_v2_o2sem.create_dataset(str(s), data=op2_normSEM[s]['O2 SEM'].to_numpy())

        supgrp_v2_iav.create_dataset(str(s), data=dop2_value[s]['iratio mean'].to_numpy())
        supgrp_v2_isd.create_dataset(str(s), data=dop2_value[s]['iratio SD'].to_numpy())
        supgrp_v2_isem.create_dataset(str(s), data=[i.s for i in op2_normSEM['set1']['iratio SEM']])

    print('saving done')
    # ------------------------------------------------------------------------------
    f.close()


def save_o2results(save_name, inp, file_meas, RoI_op, px2mm, surface, dO2_av, dO2_SD):
    # preparation
    name_files = list()
    for f in glob(file_meas + '/*R*.tif'):
        name_files.append(f.split('/')[-1])

    # ------------------------------------------------------------------------------
    # saving
    f = h5py.File(save_name, "w")

    # group  creation
    grp_header = f.create_group('header')
    grp_data = f.create_group('data')

    supgrp1 = grp_data.create_group("optode1")
    supgrp1av = supgrp1.create_group("O2 mean")
    supgrp1sd = supgrp1.create_group("O2 SD")

    supgrp2 = grp_data.create_group("optode2")
    supgrp2av = supgrp2.create_group("O2 mean")
    supgrp2sd = supgrp2.create_group("O2 SD")

    # --------------------------------------------
    # fill groups
    dt = h5py.special_dtype(vlen=str)
    grp_header.create_dataset('measurement analysed', data=np.array(name_files, dtype='O'), dtype=dt)
    grp_header.create_dataset('pixel selected', data=np.array(RoI_op))

    grp_header.create_dataset('px2mm', data=px2mm)
    v = [dO2_av[inp.split(',')[0]][inp.split(',')[1].strip()].columns.to_numpy(),
         dO2_av[inp.split(',')[0]][inp.split(',')[1].strip()].index.to_numpy()]
    grp_header.create_dataset('image size', data=np.array(v, dtype='O'), dtype=dt)

    grp_header.create_dataset('surface level', data=np.array(surface))

    # --------------------
    for k, v in dO2_av['optode1'].items():
        supgrp1av.create_dataset(str(k), data=np.array(v))
        supgrp1sd.create_dataset(str(k), data=np.array(dO2_SD['optode1'][k]))

    for k, v in dO2_av['optode2'].items():
        supgrp2av.create_dataset(str(k), data=np.array(v))
        supgrp2sd.create_dataset(str(k), data=np.array(dO2_SD['optode2'][k]))
    print('saving done')
    # ------------------------------------------------------------------------------
    f.close()


# =====================================================================================
def load_analysis_v3(dir_res):
    with h5py.File(dir_res, 'r') as f:
        # load header infos
        header = f['header']
        roi = dict()
        dic_header = dict()
        for k in header.keys():  # concentration (point) and pixel
            for ki in header[k].keys():
                if ki != 'concentration' and k != 'concentration point':
                    for v in header[k][ki].values():
                        roi[ki] = (np.array(v))
                    dic_header[k] = roi
                else:
                    conc = np.array(header[k][ki])
        dic_header['concentration'] = conc

        # --------------------------------------------------------
        # load data info
        data = f['data']
        dic_data = dict()
        for k in data.keys():
            data_v = dict()
            for ki in data[k].keys():
                data_v1 = dict()
                for en, v in enumerate(data[k][ki].values()):
                    data_v1['set{:.0f}'.format(en + 1)] = np.array(v)
                    data_v[ki] = data_v1
            dic_data[k] = data_v

        # --------------------------------------------------------
        # load fit info
        # parameters - first f then k parameter; first mean then sigma
        Fit = f['fit']
        dic_fit = dict()
        for k in Fit.keys():
            dic_values = dict()
            if 'reduced' in k:
                chi_ls = list()
                for ki in Fit[k].keys():
                    chi_ls = np.array(Fit[k][ki])
                    dic_values[ki] = chi_ls
            else:
                for ki in Fit[k].keys():
                    dic_v = dict()
                    for en, v in enumerate(Fit[k][ki].values()):
                        dic_v['set{:.0f}'.format(en + 1)] = np.array(v)
                        dic_values[ki] = dic_v
            dic_fit[k] = dic_values

        # --------------------------------------------------------
        # re-arrange data format
        dnorm = dict(map(lambda o:
                         (o, dict(map(lambda s: (s, pd.DataFrame(dic_data['normalized'][o][s], index=['iratio', 'SD'],
                                                                 columns=dic_header['concentration']).T),
                                      dic_data['normalized'][o].keys()))), dic_data['normalized'].keys()))

        dinitial = dict(map(lambda o:
                            (o, dict(map(lambda s: (s, pd.DataFrame(dic_data['averaged'][o][s], index=['iratio', 'SD'],
                                                                    columns=dic_header['concentration']).T),
                                         dic_data['averaged'][o].keys()))), dic_data['averaged'].keys()))

        return dic_header, dinitial, dnorm, dic_fit


def load_calibration_para_v1(path_calib):
    # load calibration
    dic_header, dinitial, dnorm, dic_fit = load_analysis_v3(path_calib)

    # extract fit parameter
    para = dict(map(lambda o: (o, dict(map(lambda s:
                                           (s, pd.DataFrame(dic_fit['parameter'][o][s], index=['f', 'k', 'I0'],
                                                            columns=['mean', 'SD'])), dic_fit['parameter'][o].keys()))),
                    dic_fit['parameter'].keys()))

    return para


def load_calibResults(path_calib):
    # load calibration
    dic_header, dinitial, dnorm, dic_fit = load_analysis_v3(path_calib)

    # extract fit parameter
    para = dict(map(lambda o:
                    (o, dict(map(lambda s: (s, pd.DataFrame(dic_fit['parameter'][o][s], index=['f', 'k', 'i0'],
                                                            columns=['mean', 'SD'])), dic_fit['parameter'][o].keys()))),
                    dic_fit['parameter'].keys()))
    return dic_header, dinitial, dnorm, para


def load_calibration_info(path, RoI1, height, server=True, channel=('R', 'G')):
    # red channel of calibration point as array
    dcal_R1 = dict()
    dcal_R2 = dict()
    dcal_R3 = dict()
    # green channel (G1) of calibration point as array
    dcal_G1 = dict()
    dcal_G2 = dict()
    dcal_G3 = dict()
    # concentration of calibration point (integer)
    dcal_conc1 = list()
    dcal_conc2 = list()
    dcal_conc3 = list()

    for f in glob(path + '*_{}*.tif'.format(channel[0])):
        if server is True:
            fname_R = f.split('calibration/')[1].split('.')[0]
        else:
            fname_R = f.split('calibration')[1].split('.')[0] # calibration

        if 'Cal' in fname_R:
            # green channel
            fname_G = f.split(channel[0])[0] + '{}.tif'.format(channel[1])

            # setting 1
            if 'setting1' in fname_R:
                conc = fname_R.split('_')[1]

                dcal_conc1.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R1[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G1[conc] = imG

            # setting 2
            if 'setting2' in fname_R:
                conc = fname_R.split('_')[1]
                dcal_conc2.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R2[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G2[conc] = imG

            # setting 3
            if 'setting3' in fname_R:
                conc = fname_R.split('_')[1]
                dcal_conc3.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R3[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G3[conc] = imG

    # combine settings
    dict_red = dict({'set1': dcal_R1, 'set2': dcal_R2, 'set3': dcal_R3})
    dict_green = dict({'set1': dcal_G1, 'set2': dcal_G2, 'set3': dcal_G3})
    dict_conc = dict({'set1': dcal_conc1, 'set2': dcal_conc2, 'set3': dcal_conc3})

    return dict_red, dict_green, dict_conc


def load_calibration_info_v1(path, RoI1, height, server=True, channel=('B', 'G2')):
    # red channel of calibration point as array
    dcal_R1 = dict()
    dcal_R2 = dict()
    dcal_R3 = dict()
    # green channel (G1) of calibration point as array
    dcal_G1 = dict()
    dcal_G2 = dict()
    dcal_G3 = dict()
    # concentration of calibration point (integer)
    dcal_conc1 = list()
    dcal_conc2 = list()
    dcal_conc3 = list()

    for f in glob(path + '*_{}*.tif'.format(channel[0])):
        if server is True:
            fname_R = f.split('calibration/')[1].split('.')[0]
        else:
            fname_R = 'Cal' + f.split('Cal')[1].split('.')[0] # calibration

        if 'Cal' in fname_R:
            # green channel
            fname_G = f.split(channel[0])[0] + '{}.tif'.format(channel[1])

            # setting 1
            if 'setting1' in fname_R:
                conc = fname_R.split('_')[1]

                dcal_conc1.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R1[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G1[conc] = imG

            # setting 2
            if 'setting2' in fname_R:
                conc = fname_R.split('_')[1]
                dcal_conc2.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R2[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G2[conc] = imG

            # setting 3
            if 'setting3' in fname_R:
                conc = fname_R.split('_')[1]
                dcal_conc3.append(np.int(conc.split('%')[0]))
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                # load optode into dictionary
                imarrayR = list(map(lambda r: imR_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imR = list(map(lambda r: np.stack([imarrayR[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_R3[conc] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # load optode into dictionary
                imarrayG = list(map(lambda r: imG_[RoI1[r][0][1]:RoI1[r][1][1] + 1], range(len(RoI1))))
                imG = list(map(lambda r: np.stack([imarrayG[r][n][RoI1[r][0][0]:RoI1[r][2][0] + 1]
                                                   for n in np.arange(height[r] + 1)], axis=0), range(len(RoI1))))

                # combining red-channel images
                dcal_G3[conc] = imG

    # combine settings
    dict_red = dict({'set1': dcal_R1, 'set2': dcal_R2, 'set3': dcal_R3})
    dict_green = dict({'set1': dcal_G1, 'set2': dcal_G2, 'set3': dcal_G3})
    dict_conc = dict({'set1': dcal_conc1, 'set2': dcal_conc2, 'set3': dcal_conc3})

    return dict_red, dict_green, dict_conc


def load_files(path, RoI1, height, channel=('R', 'G')):
    # red channel of calibration point as array
    dcal_R1 = dict()
    dcal_R2 = dict()
    dcal_R3 = dict()
    # green channel (G1) of calibration point as array
    dcal_G1 = dict()
    dcal_G2 = dict()
    dcal_G3 = dict()
    for f in glob(path + '*_{}*.tif'.format(channel[0])):
        fname_R = f.split('/')[-1].split('.')[0]

        if 'gradient' in fname_R:
            # green channel
            fname_G = f.split(channel[0])[0] + '{}.tif'.format(channel[1])

            # setting 1
            if 'settings1' in fname_R:
                if 'new' in f:
                    count = 'new-' + f.split('settings')[1].split('_')[0]
                else:
                    count = f.split('settings')[1].split('_')[0]
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                imR = dict()
                for o in height.keys():
                    imarrayR = imR_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imR1_ = np.stack([imarrayR[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                     axis=0)
                    imR[o] = imR1_

                # 2nd optode - combining red-channel images
                dcal_R1[count] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                imG = dict()
                for o in height.keys():
                    imarrayG = imG_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imG1_ = np.stack([imarrayG[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                     axis=0)
                    imG[o] = imG1_
                # 2nd optode - combining red-channel images
                dcal_G1[count] = imG

            # setting 2
            if 'settings2' in fname_R:
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                imR = dict()
                for o in height.keys():
                    imarrayR = imR_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imR1 = np.stack([imarrayR[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                    axis=0)
                    imR[o] = imR1
                # 2nd optode - combining red-channel images
                dcal_R2[count] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                imG = dict()
                for o in height.keys():
                    imarrayG = imG_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imG1 = np.stack([imarrayG[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                    axis=0)
                    imG[o] = imG1
                # 2nd optode - combining red-channel images
                dcal_G2[count] = imG

            # setting 3
            if 'settings3' in fname_R:
                # -----------------------------------
                # store red channel as array
                pic_R = Image.open(f)
                imR_ = np.array(pic_R)
                imR = dict()
                for o in height.keys():
                    imarrayR = imR_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imR1 = np.stack([imarrayR[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                    axis=0)
                    imR[o] = imR1
                # 2nd optode - combining red-channel images
                dcal_R3[count] = imR
                # -----------------------------------
                # store green (G1) channel as array
                pic_G = Image.open(fname_G)
                imG_ = np.array(pic_G)
                # 1st optode
                imG = dict()
                for o in height.keys():
                    imarrayG = imG_[RoI1[o][0][1]:RoI1[o][1][1] + 1]
                    imG1 = np.stack([imarrayG[n][RoI1[o][0][0]:RoI1[o][2][0] + 1] for n in np.arange(height[o] + 1)],
                                    axis=0)
                    imG[o] = imG1
                # 2nd optode - combining red-channel images
                dcal_G3[count] = imG

    # combine settings
    dict_red = dict({'set1': dcal_R1, 'set2': dcal_R2, 'set3': dcal_R3})
    dict_green = dict({'set1': dcal_G1, 'set2': dcal_G2, 'set3': dcal_G3})

    return dict_red, dict_green


def read_microsensor(file_ms, encoding='latin-1'):
    # initial inspection - how many runs, where to find data
    ls_run = list()
    with open(file_ms, 'r', encoding=encoding) as f:
        for en, line in enumerate(f.readlines()):
            if '****' in line:
                ls_run.append(en)

    ddata = dict()
    l_data = list()
    l_data1 = list()
    with open(file_ms, 'r', encoding='latin-1') as f:
        for en, line in enumerate(f.readlines()):
            if ls_run[0] - 1 <= en <= ls_run[1] + 2:
                if 'Date' in line:
                    date = line.split('\t')[1]

            if len(ls_run) > 2:
                if ls_run[1] + 3 <= en <= ls_run[2] - 2:
                    l = [i.replace(',', '.') for i in line.split('\t')]
                    l_data.append(l)
                elif en >= ls_run[3] + 3:
                    l = [i.replace(',', '.') for i in line.split('\t')]
                    l_data1.append(l)
            else:
                if ls_run[1] + 3 <= en:
                    l = [i.replace(',', '.') for i in line.split('\t')]
                    l_data.append(l)
            ddata['run1'] = l_data
            ddata['run2'] = l_data1

    # re-arrangement of data
    dic_micro = dict()
    for k in ddata.keys():
        df_ = pd.DataFrame(ddata[k])
        df_ = df_.T.set_index([0, 1]).T
        df = df_.set_index('Time')

        df_crop = df[df.columns[:2]]
        unit = df_crop.columns.levels[1][-1]
        df_crop.columns = ['Depth {}'.format(unit), 'Intensity (mV)']

        # convert index into datetime format
        index_time = [datetime.datetime.strptime(date + ' ' + t, '%d-%m-%Y %H:%M:%S') for t in df_crop.index]
        df_crop.index = index_time
        df_select = df_crop.astype(float)

        dic_micro[k] = df_select

    return dic_micro


# ====================================================================================================
def imageblur(kernel, kshape, dic_int, direction='horizontal'):
    if kernel == 'blur':
        if direction == 'horizontal':
            dst = cv2.blur(dic_int, kshape)
        elif direction == 'vertical':
            dst = cv2.blur(dic_int.T, kshape).T
        else:
            raise ValueError('define direction of kernel as either horizontal or vertical')
    elif kernel == 'filter':
        kernel = np.ones(kshape, np.float32) / (kshape[0] * kshape[1])
        if direction == 'horizontal':
            dst = cv2.filter2D(dic_int, -1, kernel)
        elif direction == 'vertical':
            dst = cv2.filter2D(dic_int.T, -1, kernel).T
        else:
            raise ValueError('define direction of kernel as either horizontal or vertical')
    elif kernel == 'gauss':
        # sigmaX and sigmaY are set as 0 --> calculation from kernel
        if direction == 'horizontal':
            dst = cv2.GaussianBlur(dic_int, kshape, 0)
        elif direction == 'vertical':
            dst = cv2.GaussianBlur(dic_int.T, kshape, 0).T
        else:
            raise ValueError('define direction of kernel as either horizontal or vertical')
    return dst


def savgol_smooth(dic_int, direction, window, polynom):
    if direction == 'horizontal':
        dst = [savgol_filter(i, window, polynom) for i in dic_int]
    elif direction == 'vertical':
        dst = np.transpose([savgol_filter(dic_int[:, i], window, polynom) for i in range(dic_int.shape[1])])
    elif direction == 'square':
        dst = sgolay2d(dic_int, window_size=window, order=polynom)
    else:
        raise ValueError('define direction of kernel as either horizontal or vertical')
    return dst


def blurimage(o, s, kernel, kshape, dint, px2mm=None, surface=None, conversion=True):
    if isinstance(dint, np.ndarray):
        data = dint
    else:
        data = dint[o][s]

    # Depth profile with (horizontal, vertical, and square) Gaussian blur for one example
    if kernel == 'savgol':
        # vertical blur
        dst_v = savgol_smooth(dic_int=data, window=kshape[1], polynom=kshape[0], direction='vertical')
        # horizontal blur
        dst_h = savgol_smooth(dic_int=data, window=kshape[1], polynom=kshape[0], direction='horizontal')
        # square blur
        dst = savgol_smooth(dic_int=data, window=kshape[1], polynom=kshape[0], direction='square')
    else:
        # vertical blur
        dst_v = imageblur(kernel=kernel, kshape=(1, kshape[0]), dic_int=data, direction='horizontal')
        # horizontal blur
        dst_h = imageblur(kernel=kernel, kshape=(kshape[0], 1), dic_int=data, direction='horizontal')
        # square blur
        dst = imageblur(kernel=kernel, kshape=kshape, dic_int=data, direction='horizontal')

    # combine all options in one dictionary
    dimages = dict({'vertical': dst_v, 'horizontal': dst_h, 'square': dst})

    # convert from px to mm
    if conversion is True:
        if px2mm is None or surface is None:
            raise ValueError('all parameter for conversion of px to mm are requires. Provide px2mm, surface parameter.')
        dimages = dict(map(lambda d: (d, px2mm_conversion(df=pd.DataFrame(dimages[d]), px2mm=px2mm,
                                                          surface=surface[int(o[-1])-1])), dimages.keys()))
    return dimages


def blurimage_df(o, kernel, kshape, dint, inorm_uncer, px2mm=None, surface=None, conversion=True):
    # split in mean and std
    image_av = np.array(list(map(lambda u: [i.n for i in inorm_uncer[u]], range(len(inorm_uncer)))))
    image_std = np.array(list(map(lambda u: [i.s for i in inorm_uncer[u]], range(len(inorm_uncer)))))

    # Depth profile with (horizontal, vertical, and square) Gaussian blur for one example
    if kernel == 'savgol':
        # vertical blur
        imgv_arr = savgol_smooth(dic_int=image_av, window=kshape[1], polynom=kshape[0], direction='vertical')
        imgvSTD_arr = savgol_smooth(dic_int=image_std, window=kshape[1], polynom=kshape[0], direction='vertical')
        # horizontal blur
        imgh_arr = savgol_smooth(dic_int=image_av, window=kshape[1], polynom=kshape[0], direction='horizontal')
        imghSTD_arr = savgol_smooth(dic_int=image_std, window=kshape[1], polynom=kshape[0], direction='horizontal')
        # square blur
        img_arr = savgol_smooth(dic_int=image_av, window=kshape[1], polynom=kshape[0], direction='square')
        imgSTD_arr = savgol_smooth(dic_int=image_std, window=kshape[1], polynom=kshape[0], direction='square')
    else:
        # vertical blur
        imgv_arr = imageblur(kernel=kernel, kshape=(1, kshape[0]), dic_int=np.array(image_av), direction='horizontal')
        imgvSTD_arr = imageblur(kernel=kernel, kshape=(1, kshape[0]), dic_int=np.array(image_std),
                                direction='horizontal')
        # horizontal blur
        imgh_arr = imageblur(kernel=kernel, kshape=(kshape[0], 1), dic_int=np.array(image_av), direction='horizontal')
        imghSTD_arr = imageblur(kernel=kernel, kshape=(kshape[0], 1), dic_int=np.array(image_std),
                                direction='horizontal')
        # square blur
        img_arr = imageblur(kernel=kernel, kshape=kshape, dic_int=np.array(image_av), direction='horizontal')
        imgSTD_arr = imageblur(kernel=kernel, kshape=kshape, dic_int=np.array(image_std), direction='horizontal')

    # combine all options in one dictionary
    dst_v = pd.DataFrame(imgv_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))
    dst_v_std = pd.DataFrame(imgvSTD_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))
    dst_h = pd.DataFrame(imgh_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))
    dst_h_std = pd.DataFrame(imghSTD_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))
    dst = pd.DataFrame(img_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))
    dst_std = pd.DataFrame(imgSTD_arr, index=np.arange(0, dint.shape[0]), columns=np.arange(0, dint.shape[1]))

    dimages = dict({'vertical': dst_v, 'horizontal': dst_h, 'square': dst})
    dimagesSTD = dict({'vertical': dst_v_std, 'horizontal': dst_h_std, 'square': dst_std})

    # convert from px to mm
    if conversion is True:
        if px2mm is None or surface is None:
            raise ValueError('all parameter for conversion of px to mm are requires. Provide px2mm, surface parameter.')
        dimages = dict(map(lambda d: (d, px2mm_conversion(df=pd.DataFrame(dimages[d]), px2mm=px2mm,
                                                          surface=surface[int(o[-1]) - 1])), dimages.keys()))
        dimagesSTD = dict(map(lambda d: (d, px2mm_conversion(df=pd.DataFrame(dimagesSTD[d]), px2mm=px2mm,
                                                             surface=surface[int(o[-1]) - 1])), dimages.keys()))
    return dimages, dimagesSTD


def blur_normIntensity(dint, I0, kshape, kernel='gauss', px2mm=None, surface=None, o=None, conversion=True):
    # determine normalized intensity including uncertainty
    i0_mp = ufloat(I0[0], I0[1])
    iratio_arr = unumpy.uarray(dint, np.array(np.zeros(shape=(dint.shape))))
    inorm_uncer = iratio_arr / i0_mp

    # ......................................................................................
    # blur image
    dimages, dimagesSTD = blurimage_df(o=o, kernel=kernel, kshape=kshape, dint=dint, inorm_uncer=inorm_uncer,
                                       px2mm=px2mm, surface=surface, conversion=conversion)

    return dimages, dimagesSTD


def O2blur_optode(inp, path_calib, kernel, kshape, px2mm, surface, depth_min, depth_max, dint_ch1, dint_ch2=None,
                  blur_pos='ratio'):
    # preparation
    o = inp.split(',')[0]
    s = inp.split(',')[1].strip()

    # load calibration
    calib_info = load_calibration_para_v1(path_calib=path_calib)
    para = calib_info[o][s]

    # -------------------------------------------
    # blur images
    if blur_pos == 'norm':
        dimages, dimagesSTD = blur_normIntensity(dint=dint_ch1[o][s], I0=para.loc['I0'].to_numpy(), kernel=kernel,
                                                 kshape=kshape, px2mm=px2mm, surface=surface, o=o, conversion=True)
    else:
        dblur_ch1 = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch1, px2mm=px2mm, surface=surface)
        if blur_pos == 'ratio':
            dimages = dblur_ch1
        elif blur_pos == 'single':
            # blur individual color channels, then determine ratiometric intensity
            dgreen_blur = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch2, px2mm=px2mm, surface=surface)
            dimages = dict(map(lambda ax: (ax, dblur_ch1[ax] / dgreen_blur[ax]), dblur_ch1.keys()))
        else:
            raise ValueError('select a valid argument for int_type. Chose either norm, ratio, or single:'
                             'norm ... blur normalized intensity'
                             'ratio ... blur ratiometric intensity'
                             'single ... blur individual color channels')

    # crop to image frame of interest
    dimg = dict(map(lambda d: (d, dimages[d].loc[depth_min:depth_max, :] if dimages[d].empty is False else None),
                dimages.keys()))
    if blur_pos == 'norm':
        dimg_std = dict(map(lambda d:
                            (d, dimagesSTD[d].loc[depth_min:depth_max, :] if dimagesSTD[d].empty is False else None),
                            dimagesSTD.keys()))
    else:
        dimg_std = dict(map(lambda d: (d, None), dimages.keys()))

    # -------------------------------------------
    # determine O2 concentration
    dO2_calc_ = dict()
    for h in dimages.keys():
        if dimages[h].empty is True:
            pass
        else:
            dO2_calc = O2_analysis_area(para=para, iratio=dimg[h], iratio_std=dimg_std[h], int_type=blur_pos)
            dO2_calc_[h] = dO2_calc

    # split in mean and SD
    dO2_optode = dict()
    for d in dO2_calc_.keys():
        dO2op_av = pd.concat(dict(map(lambda c:
                                      (c, pd.DataFrame([i.n for i in dO2_calc_[d][c]], index=dO2_calc_[d].index)),
                                      dO2_calc_[d].columns)), axis=1, ignore_index=True)
        dO2op_av.columns = dO2_calc_[d].columns
        dO2_optode[d] = dO2op_av

    return dO2_optode


def postprocessing_v1(dO2_calc, px2mm, surface, split=True, vmin=-50, vmax=150):
    print('post-processing...')
    # remove obvious outlier (min_, max_) and convert px to mm
    dO2 = postprocessing(O2_calc=dO2_calc, px2mm=px2mm, baseline=surface, min_=vmin, max_=vmax)

    # split ufloat into mean and sd for visualization
    if split is True:
        dO2_av, dO2_SD = split2statics(dO2)
    else:
        dO2_av, dO2_SD = None, None

    return dO2, dO2_av, dO2_SD


def postprocessing(px2mm, baseline, O2_calc, min_=-50, max_=150):
    # convert array into dataframe
    dfO2_calc = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(O2_calc[o][s])), O2_calc[o].keys()))),
                          O2_calc.keys()))

    # remove obvious outlier x< -50 or x > 1000 %air
    for o in dfO2_calc.keys():
        for s in dfO2_calc[o].keys():
            # run 2 - new gradient
            dfO2_calc[o][s][dfO2_calc[o][s] < min_] = np.nan
            dfO2_calc[o][s][dfO2_calc[o][s] > max_] = np.nan

    # convert px-> mm
    for en, o in enumerate(dfO2_calc.keys()):
        for s in dfO2_calc[o].keys():
            # run1 - gradient
            ind_new = dfO2_calc[o][s].index.to_numpy() / px2mm - baseline[en]
            col_new = dfO2_calc[o][s].columns.to_numpy() / px2mm

            # rename index and columns for optode1 - mean and SD
            dfO2_calc[o][s].index = ind_new
            dfO2_calc[o][s].columns = col_new

    return dfO2_calc


# ----------------------------------------------------------
def O2concentration_lp(para, ls_lw, ddlp, ddlp_std=None, int_type='norm'):
    # determine O2 concentration
    if ddlp_std is None:
        dO2 = dict(map(lambda d:
                       (d, dict(map(lambda lw_: (lw_, pd.concat([O2_analysis_area(para=para, int_type=int_type,
                                                                                  iratio=ddlp[d][lw_][c],
                                                                                  iratio_std=None)
                                                                 for c in ddlp[d][lw_].columns],
                                                                axis=1).fillna(limit=5, method='ffill')
                       if ddlp[d][lw_] is not None else None), ddlp[d].keys()))), ddlp.keys()))
    else:
        dO2 = dict(map(lambda d:
                       (d, dict(map(lambda lw_: (lw_, pd.concat([O2_analysis_area(para=para, int_type=int_type,
                                                                                  iratio=ddlp[d][lw_][c],
                                                                                  iratio_std=ddlp_std[d][lw_][c])
                                                                 for c in ddlp[d][lw_].columns],
                                                                axis=1).fillna(limit=5, method='ffill')
                       if ddlp[d][lw_] is not None else None), ddlp[d].keys()))), ddlp.keys()))

    # averaging for mean and SD
    dO2_depth = dict()
    for d in dO2.keys():
        ddO2_dp = dict()
        for lw_ in ls_lw:
            d_av, d_sd = dict(), dict()
            if dO2[d][lw_] is not None:
                for c in dO2[d][lw_].columns:
                    d_av_ = pd.DataFrame([i.n for i in dO2[d][lw_][c].to_numpy()], index=dO2[d][lw_].index,
                                         columns=['mean'])
                    d_sd_ = pd.DataFrame([i.s for i in dO2[d][lw_][c].to_numpy()], index=dO2[d][lw_].index,
                                         columns=['SD'])
                    d_av[c], d_sd[c] = d_av_, d_sd_

                dO2_dp = pd.concat([pd.concat(d_av, axis=1, ignore_index=True).mean(axis=1),
                                    pd.concat(d_sd, axis=1, ignore_index=True).mean(axis=1)], axis=1)
                dO2_dp.columns = ['mean', 'SD']
                ddO2_dp[lw_] = dO2_dp
            else:
                ddO2_dp[lw_] = None
        dO2_depth[d] = ddO2_dp

    return dO2_depth


def O2_lineprofile_compare_v1(inp, surface, kernel, kshape, px2mm, lp, ls_lw, path_calib, dint_ch1, dint_ch2=None,
                              blur_type='ratio'):
    # preparation
    o, s = inp.split(',')[0], inp.split(',')[1].strip()

    # load calibration
    calib_info = load_calibration_para_v1(path_calib=path_calib)
    para = calib_info[o][s]

    # blur images
    dblur_ch1 = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch1, px2mm=px2mm, surface=surface)
    if blur_type == 'ratio':
        dimages = dblur_ch1
    else:
        # blur individual color channels, then determine ratiometric intensity
        dgreen_blur = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch2, px2mm=px2mm, surface=surface)
        dimages = dict(map(lambda ax: (ax, dblur_ch1[ax] / dgreen_blur[ax]), dblur_ch1.keys()))

    # crop to image width of interest
    ddlp = dict(map(lambda d: (d, dict(map(lambda lw_: (lw_, line_profile_v1(df=dimages[d], lw=lw_, lp=lp[0])), ls_lw))),
                    dimages.keys()))

    # determine O2 concentration for line profile
    dO2_lp = O2concentration_lp(para=para, ddlp=ddlp, ls_lw=ls_lw)

    return dO2_lp


def O2_lineprofile_compare_v2(inp, surface, kernel, kshape, px2mm, lp, ls_lw, path_calib, dint_ch1, dint_ch2=None,
                              blur_pos='norm'):
    # preparation
    o = inp.split(',')[0]
    s = inp.split(',')[1].strip()

    # load calibration
    calib_info = load_calibration_para_v1(path_calib=path_calib)
    para = calib_info[o][s]

    # blur images
    if blur_pos == 'norm':
        dimages, dimagesSTD = blur_normIntensity(dint=dint_ch1[o][s], I0=para.loc['I0'].to_numpy(), kernel=kernel,
                                                 kshape=kshape, px2mm=px2mm, surface=surface, o=o, conversion=True)
    else:
        dblur_ch1 = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch1, px2mm=px2mm, surface=surface)
        if blur_pos == 'ratio':
            dimages = dblur_ch1
        elif blur_pos == 'single':
            # blur individual color channels, then determine ratiometric intensity
            dgreen_blur = blurimage(o=o, s=s, kernel=kernel, kshape=kshape, dint=dint_ch2, px2mm=px2mm, surface=surface)
            dimages = dict(map(lambda ax: (ax, dblur_ch1[ax] / dgreen_blur[ax]), dblur_ch1.keys()))
        else:
            raise ValueError('select a valid argument for int_type. Chose either norm, ratio, or single:'
                             'norm ... blur normalized intensity'
                             'ratio ... blur ratiometric intensity'
                             'single ... blur individual color channels')

    # crop to image width of interest
    ddlp = dict(map(lambda d: (d, dict(map(lambda lw_: (lw_, line_profile_v1(df=dimages[d], lw=lw_, lp=lp[0])),
                                           ls_lw))), dimages.keys()))

    if blur_pos == 'norm':
        ddlp_std = dict(map(lambda d:
                            (d, dict(map(lambda lw_: (lw_, line_profile_v1(df=dimagesSTD[d], lw=lw_, lp=lp[0])),
                                         ls_lw))), dimages.keys()))
    else:
        ddlp_std = None

    # determine O2 concentration for line profile
    dO2_lp = O2concentration_lp(para=para, ls_lw=ls_lw, ddlp=ddlp, ddlp_std=ddlp_std, int_type=blur_pos)
    return dO2_lp

