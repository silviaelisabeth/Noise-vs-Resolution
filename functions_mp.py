__author__ = 'Silvia E Zieger'
__project__ = 'noise vs resolution'

"""Copyright 2020. All rights reserved.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable 
for any damages arising from the use of this software.
Permission is granted to anyone to use this software within the scope of evaluating mutli-analyte sensing. No permission
is granted to use the software for commercial applications, and alter it or redistribute it.

This notice may not be removed or altered from any distribution.
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
from lmfit import Model
from mcerp import *
from uncertainties import *
import datetime
import h5py
import os
from PIL import Image

import functions_NoiseStudy as noise

# global variables
col = ['#14274e', '#f6830f', '#bb2205']
mark = ['o', 'd']
fs = 13


# =========================================================================================
def define_output():
    # actual evaluation time
    now = datetime.datetime.now()

    # define output folder
    save_dir_plots = 'plots/' + now.strftime("%Y%m%d")
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    save_dir_res = 'Results/' + now.strftime("%Y%m%d")
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    return now, save_dir_res


def load_calibration_img(path, RoI_op, arg):
    # load images from folder

    # load test image to define cropping area of the optode
    test1 = path + 'Cal_0%_setting1_0000_R.tif'
    imT1 = np.array(Image.open(test1))
    test2 = path + 'Cal_0%_setting2_0000_R.tif'
    imT2 = np.array(Image.open(test2))
    test3 = path + 'Cal_0%_setting3_0000_R.tif'
    imT3 = np.array(Image.open(test3))

    # ----------------------------------------------------------------------------------------------------------------
    # Crop image to optode area - anti-clockwise starting at the top left (index x columns)
    # RoI for different settings
    height = list(map(lambda n: (RoI_op[n][1][1] - RoI_op[n][0][1]), range(len(RoI_op))))

    # collect information
    # in imageJ the counting is (x,y,z) with x 0-2599 and y 0-1731
    # in Python the counting is (ind, column, z) with ind 0-1732 and col 0-2601
    # np.arrays are 1 x1 (col x rows) too huge compared to imageJ
    dict_red, dict_green, dict_conc = noise.load_calibration_info(path, RoI_op, height, server=False)
    print('number of calibration points ', len(dict_red['set1'].keys()))

    # ----------------------------------------------------------------------------------------------------------------
    # calculating the ratio R/G of the whole optode
    if arg['ratiometric'] == 'ratio' or arg['ratiometric'] == None:
        dratio1 = dict()
        dratio2 = dict()
        dratio3 = dict()
        for c in dict_conc['set1']:
            dratio1[c] = [dict_red['set1'][str(c) + '%'][n] / dict_green['set1'][str(c) + '%'][n]
                          for n in range(len(dict_red['set1'][str(c) + '%']))]
            dratio2[c] = [dict_red['set2'][str(c) + '%'][n] / dict_green['set2'][str(c) + '%'][n]
                          for n in range(len(dict_red['set2'][str(c) + '%']))]
            dratio3[c] = [dict_red['set3'][str(c) + '%'][n] / dict_green['set3'][str(c) + '%'][n]
                          for n in range(len(dict_red['set3'][str(c) + '%']))]
    elif arg['ratiometric'] == 'red':
        dratio1 = dict()
        dratio2 = dict()
        dratio3 = dict()
        for c in dict_conc['set1']:
            dratio1[c] = [dict_red['set1'][str(c) + '%'][n] for n in range(len(dict_red['set1'][str(c) + '%']))]
            dratio2[c] = [dict_red['set2'][str(c) + '%'][n] for n in range(len(dict_red['set2'][str(c) + '%']))]
            dratio3[c] = [dict_red['set3'][str(c) + '%'][n] for n in range(len(dict_red['set3'][str(c) + '%']))]
    else:
        dratio1, dratio2, dratio3 = None, None, None
    return dratio1, dratio2, dratio3


def normalization(dratio1, dratio2, dratio3, RoI_op):
    dic_optode = dict()
    for o in range(len(RoI_op)):
        opt_set1_norm = dict(map(lambda c: (c, dratio1[0][o] / dratio1[c][o]), dratio1.keys()))
        opt_set2_norm = dict(map(lambda c: (c, dratio2[0][o] / dratio2[c][o]), dratio2.keys()))
        opt_set3_norm = dict(map(lambda c: (c, dratio3[0][o] / dratio3[c][o]), dratio3.keys()))
        # combine all settings in a dictionary
        opt_norm = dict({'set1': opt_set1_norm, 'set2': opt_set2_norm, 'set3': opt_set3_norm})

        # combine everything (all optodes) in a dictionary
        dic_optode[o] = opt_norm
    return dic_optode


def rawdata2optode(dratio1, dratio2, dratio3, pos=0):
    # extract information about optode on position x
    opt = dict({'set1': dict(map(lambda c: (c, dratio1[c][pos]), dratio1.keys())),
                'set2': dict(map(lambda c: (c, dratio2[c][pos]), dratio2.keys())),
                'set3': dict(map(lambda c: (c, dratio3[c][pos]), dratio3.keys()))})
    return opt


def rearrange_dict(opt_norm, RoI_op):
    # re-arrangement of data - each pixel shows calibration
    opt_norm_re = dict(map(lambda o: (o, dict(map(lambda s: (s, noise.rearrange_optode(opt_norm[o][s])),
                                                  opt_norm[o].keys()))), range(len(RoI_op))))
    return opt_norm_re


def preparation_saving_mp(dratio1, dratio2, dratio3, opt_norm, opt_norm_re, res, nRoI):
    # prepare pixel information (min, max as list for height and width)
    pixel = list()
    for o in range(len(nRoI)):
        px = opt_norm_re[o]['set1'][1].shape
        # output as tuple for each optode
        pixel.append(([0, px[0]], [0, px[1]]))

    # ----------------------------------------------------
    # prepare raw data
    opt = dict(map(lambda o: (o, rawdata2optode(dratio1, dratio2, dratio3, pos=o)), range(len(nRoI))))

    optode_out = dict(map(lambda o: (o, dict(map(lambda s: (s, np.array([i for i in opt[s].values()])), opt.keys()))),
                          range(len(nRoI))))
    # ----------------------------------------------------
    # normalized data
    if opt_norm:
        opt_norm_out = dict(map(lambda o: (o, dict(map(lambda s: (s, np.array([opt_norm[o][s][c]
                                                                               for c in opt_norm[o][s].keys()])),
                                                       opt_norm[o].keys()))), range(len(nRoI))))
    else:
        opt_norm_out = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                       optode_out[o].keys()))), range(len(nRoI))))
    print(152, '---------------------------------------')
    print(opt_norm_out)
    # ----------------------------------------------------
    # best fit
    if res:
        bestFit = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame([res[o][s][px][1].best_fit
                                                                              for px in res[o][s].keys()],
                                                      index=res[o][s].keys())), res[o].keys()))), range(len(nRoI))))
    else:
        bestFit = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                  optode_out[o].keys()))), range(len(nRoI))))
    print(bestFit)
    # ----------------------------------------------------
    # fit parameter
    # create dic of dataframe for each set and pixel to combine the fit parameter f and k
    # optode 1
    if res:
        params = dict(map(lambda o: (o,
                                     dict(map(lambda s:
                                              (s, pd.concat([pd.DataFrame([(res[o][s][k][2]['f'][0], res[o][s][k][2]['f'][1])
                                                                           for k in res[o][s].keys()], index=res[o][s].keys(),
                                                                          columns=['mean f', 'SD f']),
                                                             pd.DataFrame([(res[o][s][k][2]['k'][0], res[o][s][k][2]['k'][1])
                                                                           for k in res[o][s].keys()], index=res[o][s].keys(),
                                                                          columns=['mean k', 'SD k'])], axis=1)),
                                              res[o].keys()))), range(len(nRoI))))
    else:
        params = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(2, 2)),
                                                                            columns=['mean', 'SD'], index=['f', 'k'])),
                                                 optode_out[o].keys()))), range(len(nRoI))))
    print(params)
    # reduced chi-square
    # dictionary of chi-squares (index = pixel) for each optode
    if res:
        redchi = dict(map(lambda o: (o,
                                     dict(map(lambda s: (s, pd.DataFrame([res[o][s][k][2]['red chi-square']
                                                                          for k in res[o][s].keys()],
                                                                         index=res[o][s].keys())), res[o].keys()))),
                          range(len(nRoI))))
    else:
        redchi = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                 optode_out[o].keys()))), range(len(nRoI))))
    print(redchi)
    return pixel, optode_out, opt_norm_out, params, redchi, bestFit


def _preparationSaving_mp(dratio1, dratio2, dratio3, opt_norm, opt_norm_re, res, nRoI):
    # prepare pixel information (min, max as list for height and width)
    pixel = list()
    for o in range(len(nRoI)):
        px = opt_norm_re[o]['set1'][1].shape
        # output as tuple for each optode
        pixel.append(([0, px[0]], [0, px[1]]))

    # ----------------------------------------------------
    # prepare raw data (including i0)
    opt = noise._split2optode(dratio1=dratio1, dratio2=dratio2, dratio3=dratio3, nRoI=nRoI)
    # opt = dict(map(lambda o: (o, rawdata2optode(dratio1, dratio2, dratio3, pos=o)), range(len(nRoI))))

    optode_out = dict(map(lambda o: (o, dict(map(lambda s: (s, np.array([i for i in opt[s].values()])), opt.keys()))),
                          range(len(nRoI))))
    print(211, 'raw data')
    # ----------------------------------------------------
    # normalized data
    if opt_norm:
        opt_norm_out = dict()
        for o in range(len(nRoI)):
             arr_norm = noise._normalize_mp(opt=opt_norm[o])
             opt_norm_out[o] = arr_norm
    else:
        opt_norm_out = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                       optode_out[o].keys()))), range(len(nRoI))))
    print(222, 'normalized data')
    # ----------------------------------------------------
    # best fit
    if res:
        bestFit = dict()
        for o in range(len(nRoI)):
            bestFit_opt = noise._bestFit_mp(res=res[o])
            bestFit[o] = bestFit_opt
    else:
        bestFit = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                  optode_out[o].keys()))), range(len(nRoI))))
    print(233, 'bestFit data')
    # ----------------------------------------------------
    # fit parameter - create dic of dataframe for each set and pixel to combine the fit parameter f and k
    if res:
        params = dict()
        for o in range(len(nRoI)):
            params_opt = noise._paraFit_mp(res=res[o])
            params[o] = params_opt
    else:
        params = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(2, 2)),
                                                                            columns=['mean', 'SD'], index=['f', 'k'])),
                                                 optode_out[o].keys()))), range(len(nRoI))))
    print(245, 'fitParameter data')
    # ----------------------------------------------------
    # reduced chi-square - dictionary of chi-squares (index = pixel) for each optode
    if res:
        redchi = dict()
        for o in range(len(nRoI)):
            chi_out = noise._chi_mp(res=res[o])
            redchi[o] = chi_out
    else:
        redchi = dict(map(lambda o: (o, dict(map(lambda s: (s, pd.DataFrame(np.zeros(shape=(1, 1)))),
                                                 optode_out[o].keys()))), range(len(nRoI))))
    print(256, 'chi data')

    return pixel, optode_out, opt_norm_out, params, redchi, bestFit


def saving_results_mp(save_name, conc, pixel, optode_out, opt_norm, params, redchi, bestFit, RoI_op):
    # [actual saving]
    f = h5py.File(save_name, "w")

    # ------------------------------------------------------------------------------
    # [group creation]
    # header
    grp_header = f.create_group('header')
    subgrp_conc = grp_header.create_group("concentration point")
    subgrp_px = grp_header.create_group("pixel")

    # data group
    grp_data = f.create_group("data")
    subgrp_raw = grp_data.create_group('raw')
    dsubgrp_raw = dict()
    for o in range(len(RoI_op)):
        g_raw = subgrp_raw.create_group('optode-' + str(o + 1))
        dsubsub_raw = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_raw = g_raw.create_group('set' + str(s + 1))
            dsubC_raw = dict()
            for c in range(len(optode_out[0][o][s])):
                sub_c_raw = sub_g_raw.create_group(str(c))
                dsubC_raw[c] = sub_c_raw
            dsubsub_raw[s] = dsubC_raw
        dsubgrp_raw[o] = dsubsub_raw

    subgrp_norm = grp_data.create_group('normalized')
    dsubgrp_norm = dict()
    for o in range(len(RoI_op)):
        g_norm = subgrp_norm.create_group('optode-' + str(o + 1))
        dsubsub_norm = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_norm = g_norm.create_group('set' + str(s + 1))
            dsubsub_norm[s] = sub_g_norm
        dsubgrp_norm[o] = dsubsub_norm

    # ---------------------------------------------------------------------
    # group related to fit process
    grp_fit = f.create_group("fit")
    subgrp_param = grp_fit.create_group("parameter")
    dsubgrp_para = dict()
    for o in range(len(RoI_op)):
        g_para = subgrp_param.create_group('optode-' + str(o + 1))
        dsubsub_para = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_para = g_para.create_group('setting-' + str(s + 1))
            dsubsub_para[s] = sub_g_para
        dsubgrp_para[o] = dsubsub_para

    subgrp_I0 = grp_fit.create_group("I0")
    dsubgrp_I0 = dict()
    for o in range(len(RoI_op)):
        g_i0 = subgrp_I0.create_group('optode-' + str(o + 1))
        dsubsub_i0 = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_i0 = g_i0.create_group('setting-' + str(s + 1))
            dsubsub_i0[s] = sub_g_i0
        dsubgrp_I0[o] = dsubsub_i0

    subgrp_chi = grp_fit.create_group("reduced chi-square")
    dsubgrp_chi = dict()
    for o in range(len(RoI_op)):
        g_chi = subgrp_chi.create_group('optode-' + str(o + 1))
        dsubsub_chi = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_chi = g_chi.create_group('setting-' + str(s + 1))
            dsubsub_chi[s] = sub_g_chi
        dsubgrp_chi[o] = dsubsub_chi

    subgrp_values = grp_fit.create_group("best fit")
    dsubgrp_values = dict()
    for o in range(len(RoI_op)):
        g_values = subgrp_values.create_group('optode-' + str(o + 1))
        dsubsub_chi = dict()
        for s in range(len(optode_out[0][o])):
            sub_g_values = g_values.create_group('setting-' + str(s + 1))
            dsubsub_chi[s] = sub_g_values
        dsubgrp_values[o] = dsubsub_chi

    # ------------------------------------------------------------------------------
    # [fill groups]
    subgrp_conc.create_dataset('concentration', data=conc[0]['set1'][0])

    for o in range(len(RoI_op)):
        subgrp_px.create_dataset('optode-{} - px_height'.format(str(o+1)), data=pixel[o][1])
        subgrp_px.create_dataset('optode-{} - px_width'.format(str(o+1)), data=pixel[o][0])
    print(345)
    # subgroup - raw data
    for o in range(len(RoI_op)): # optode
        for s in range(len(optode_out[0][o])): # setting
            print(o, s)
            print(optode_out[0][o][s].keys())
            for k, v in enumerate(optode_out[0][o][s].items()): # concentration
                print(dsubgrp_raw[o][s].keys())
                dsubgrp_raw[o][s][k].create_dataset(str(v[0]), data=np.array(v[1]))
    print(351)
    # subgroup - normalized data
    for o in range(len(RoI_op)):
        for s, v in enumerate(opt_norm[o].items()):
            dsubgrp_norm[o][s].create_dataset(str(v[0]), data=np.array(v[1][1]))
    print(356)
    # subgroup - fit parameters
    for o in range(len(RoI_op)):
        for s, v in enumerate(params[o].items()):
            dsubgrp_para[o][s].create_dataset(str(v[0]), data=np.array(v[1][1]))
    print(361)
    # subgroup - I0
    for o in range(len(RoI_op)):
        for s in range(len(optode_out[0][o])):
            v = 'set' + str(s)
            dsubgrp_I0[o][s].create_dataset(v, data=np.array(optode_out[0][o][s][0]))
    print(367)
    # subgroup - reduces chi-square
    for o in range(len(RoI_op)):
        for s, v in enumerate(redchi[o].items()): # k := set, v:= dataframe (chi-square optode1, index = pixel)
            dsubgrp_chi[o][s].create_dataset(str(v[0]), data=np.array(v[1][1][0].to_numpy()))
    print(372)
    # subgroup - fit values
    for o in range(len(RoI_op)):
        for s, v in enumerate(bestFit[o].items()):
            dsubgrp_values[o][s].create_dataset(str(v[0]), data=np.array(v[1][1]))

    # ------------------------------------------------------------------------------
    f.close()
    print('saving done')


# =========================================================================================
def _main_calibration_px(path, RoI_op, arg, saving=False):
    """
    O2 calibration in each pixel
    :return:
    """
    # define output folder
    now, save_dir_res = define_output()

    # ------------------------------------------------------------------------------------
    # load red / green channel for calibration and calculate ratio = red / green
    dratio1, dratio2, dratio3 = load_calibration_img(path=path, RoI_op=RoI_op, arg=arg)

    # ------------------------------------------------------------------------------------
    # normalization
    dic_opt_norm = normalization(dratio1=dratio1, dratio2=dratio2, dratio3=dratio3, RoI_op=RoI_op)

    # -----------------------------
    # re-arrangement of dict
    opt_norm_re = rearrange_dict(opt_norm=dic_opt_norm, RoI_op=RoI_op)

    # ------------------------------------------------------------------------------------
    # multiprocessing for calibration - nested for loop!!
    dic_res = dict()
    for o in range(len(RoI_op)):
        a, b = int(len(opt_norm_re[o]['set1'][1])), int(len(opt_norm_re[0]['set1'][1][0]))
        print('number of pixels for calibration', a, 'x', b)
        res = noise._simplifiedSV_mp(opt=opt_norm_re[o], px_size=(a, b))
        dic_res[o] = res
    print('calibration done')

    # ------------------------------------------------------------------------------------
    # saving all information
    # store the following information in hdf5 file:
    eval_time = now.strftime("%Y%m%d-%H%M")
    save_name = save_dir_res + '/' + eval_time + '_calibration_pixel-by-pixel.hdf5'

    # [preparing each optode for saving - parallelize code]
    [pixel, optode_out, opt_norm_out, para, redchi,
     bestFit] = _preparationSaving_mp(dratio1=dratio1, dratio2=dratio2, dratio3=dratio3, opt_norm=dic_opt_norm,
                                      opt_norm_re=opt_norm_re, res=dic_res, nRoI=RoI_op)
    print('preparation done')
    if saving is True:
        saving_results_mp(save_name=save_name, conc=opt_norm_re, pixel=pixel, optode_out=optode_out, RoI_op=RoI_op,
                          opt_norm=opt_norm_out, params=para, redchi=redchi, bestFit=bestFit)

    return save_name, pixel, opt_norm_re, optode_out, opt_norm_out, para, redchi, bestFit


# =========================================================================================
# # CONSOLE WORK
# path = 'calibration/'
#path = 'E:/04measurementData/20201127_Noise-vs-resolution-paper/Klaus_Optode_noise_study_26-11-2020/calibration/'
#RoI_op = [[(850, 424), (850, 777), (1211, 777), (1211, 424)]] # only optode1
#arg = dict({'ratiometric': 'ratio'})

#if __name__ == '__main__':
#    opt_norm_re = _main_calibration_px(path, RoI_op, arg=arg, saving=True)
