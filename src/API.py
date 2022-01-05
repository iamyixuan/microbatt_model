import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as t
import tensorflow.keras as k
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import matplotlib.cm as cm


def make_prediction_alpha(mse_pred, mape_pred, bin_scores_mse, bin_scores_mape):
    bins = bins = [0, 3, 6, 9, 12, 15]
    bin_idx_mse = np.digitize(mse_pred, bins, right=False)
    bin_idx_mape = np.digitize(mape_pred, bins, right=False)
    final_pred = np.zeros(mse_pred.shape)
    for i in range(len(bin_idx_mse)):
        if bin_idx_mse[i] == bin_idx_mape[i]:
            if bin_idx_mse[i] == 1 or bin_idx_mse[i] == 2:
                final_pred[i] = mape_pred[i]
            elif bin_idx_mse[i] == 3 or bin_idx_mse[i] == 4 or bin_idx_mse[i] == 5 or bin_idx_mse[i] == 6:
                final_pred[i] = mse_pred[i]
        else:
            print('triger')
            bin_num_1 = bin_idx_mse[i]
            bin_num_2 = bin_idx_mape[i]
            print(bin_num_1, bin_num_2)

            mse_score_bin1 = bin_scores_mse[bin_num_1 - 1]
            mse_score_bin2 = bin_scores_mse[bin_num_2 - 1]

            mape_score_bin1 = bin_scores_mape[bin_num_1 - 1]
            mape_score_bin2 = bin_scores_mape[bin_num_2 - 1]

            if mse_score_bin1 < mape_score_bin1 and mse_score_bin2 < mape_score_bin2:
                print('triger1')

                final_pred[i] = mse_pred[i]
            elif mse_score_bin1 > mape_score_bin1 and mse_score_bin2 > mape_score_bin2:
                print('triger2')
                final_pred[i] = mape_pred[i]
            else:
                diff_bin1 = mse_score_bin1 - mape_score_bin1
                diff_bin2 = mse_score_bin2 - mape_score_bin2
                if np.abs(diff_bin1) > np.abs(diff_bin2):
                    
                    if diff_bin1 > 0:
                        final_pred[i] = mape_pred[i]
                    else:
                        final_pred[i] = mse_pred[i]
                else:
                    if diff_bin2 > 0:
                        final_pred[i] = mape_pred[i]
                    else:
                        final_pred[i] = mse_pred[i]
    return final_pred

"""
User input: 
cell potential sequences (cut off 2.2 V)
capacity seqences or time sequences
Current density
Power density
Energy density
"""



def prediction(cell_pot_seq, capacity_seq, current_density_list, power_density_list, energy_density_list):
    """
    Predicting microstructure property of a battery: Bruggeman's exponent and shape factor.
    cell_pot_seq: shape: (6, seq_len); cell potential sequences corresponding to 6 current densities.
    capacity_seq: shape: (6, seq_len); capacity sequences corresponding to 6 current densities.
    current_density_seq: current density values of which the order needs to be consistent with other variables. It needs to contain
    power_density_list: length: 6; power density values corresponding to the 6 current densities. 
    energy_density_list: length:6; energy density values for the 6 current densities.
    """

    def weighted_MSE(yTrue, yPred, w_alpha=0.1):

        return K.sum(w_alpha*(yTrue[:,0]-yPred[:,0])**2 + (1-w_alpha)*(yTrue[:,1]-yPred[:,1])**2)

    def r2(y_true, y_pred):
        tol_sum_sq = K.sum((y_true - K.mean(y_true))**2)
        residual_sum_sq = K.sum((y_true - y_pred)**2)
        r2 = 1 - (residual_sum_sq/tol_sum_sq)
        return r2
    #load seq info
    def get_time(capacity, current):
        return 600*capacity/current


    color_current = np.array([1.75, 8.75, 17.5, 35, 52.5, 70])

    colors = cm.rainbow(np.linspace(0, 1, 6))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axis('off')
    for i in range(6):
        color_index = np.argmin(np.abs(current_density_list[i] - color_current))
        #capacity_tmp = time[i]/600 * currents[i]
        ax.plot(capacity_seq[i], cell_pot_seq[i], color=colors[color_index])
        ax.set_xlim(0, 2.0)
        ax.set_ylim(2.0, 4.7)

    if not os.path.exists("./test_tmp/"):
        os.makedirs("./test_tmp/")
    fig.savefig('./test_tmp/test_fig_norm_trun_capacity.png', format='png', dpi=100)

    im_frame = cv2.imread('./test_tmp/test_fig_norm_trun_capacity.png')
    im_frame = cv2.resize(im_frame, (64, 48), interpolation=cv2.INTER_AREA)
    np_frame = np.array(im_frame)
    np_frame = np.expand_dims(np_frame, axis=0)


    extra_info = np.concatenate([energy_density_list.reshape(-1 ,1), power_density_list.reshape(-1, 1)], axis=1)
    extra_info = extra_info.reshape(-1, )
    # make sure the extra info input order is consistent with the training


    # extra_info_t = np.load('../data/cell_pot_capacity_EP/0.5_3.0.npy', 'r', True).reshape(1, -1)
    # sorted_extra = np.argsort(extra_info_t).reshape(-1, )
    # sorted_test = np.sort(extra_info).reshape(-1, )

    # temp = np.zeros(len(sorted_extra))
    # for i, order in enumerate(sorted_extra):
    #     temp[order] = sorted_test[i]
    # print(temp.shape)
    
    model = k.models.load_model('/content/microbatt_model/src/models/six_curve_model_cell_pot_capacity_3_30.h5', custom_objects={'weighted_MSE': weighted_MSE, 'r2': r2})
    model_mape = k.models.load_model('/content/microbatt_model/src/models/mape_model.h5', custom_objects={'r2': r2})
    
    pred_mse = model.predict([np_frame, extra_info.reshape(1, -1)])
    pred_mape = model_mape.predict([np_frame, extra_info.reshape(1, -1)])

    print("MSE Predicted Alpha is {:.2f} and K is {:.2f}".format(pred_mse[0][0], pred_mse[0][1]))
    print("MAPE Predicted Alpha is {:.2f} and K is {:.2f}".format(pred_mape[0][0], pred_mape[0][1]))

    mse_scores = [12.202474474906921, 2.754971571266651, 0.9775176644325256, 0.6340787746012211, 0.8929776959121227, 0.33431430347263813]
    mape_scores = [6.173022091388702, 1.1646689847111702, 1.1303179897367954, 0.8545964024960995, 1.0012668557465076, 0.40570483542978764]

    final_pred = make_prediction_alpha(np.array([pred_mse[0][0]]), np.array([pred_mape[0][0]]), mse_scores, mape_scores)
    print("Final predicted Alpha is {:.2f} and K is {:.2f}".format(float(final_pred), pred_mse[0][1]))

    return final_pred, pred_mse[0][1]

if __name__ == "__main__":
    test_data_dir = '../data/more_testing_raw/set4/'
    files = glob.glob(test_data_dir +'*.csv')
    currents = []
    cell_pot = []
    capacity = []
    energy = []
    power = []
    for f in files:
        temp = pd.read_csv(f)
        currents.append(temp['Curr den (A/m^2)'].values[0])
        cell_pot.append(temp['Cell pot(V)'].values)
        capacity.append(temp['Capacity'].values)
        energy.append(temp['Energy'].values[0])
        power.append(temp['Power'].values[0])

    idx_ = np.argsort(currents)

    prediction(np.array(cell_pot)[idx_], np.array(capacity)[idx_], np.array(currents)[idx_], np.array(power)[idx_], np.array(energy)[idx_])
