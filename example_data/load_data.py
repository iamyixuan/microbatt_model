import glob
import numpy as np
import pandas as pd

def load_example_data(None):
    test_data_dir = './set0/'
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
    return np.array(cell_pot)[idx_], np.array(capacity)[idx_], np.array(currents)[idx_], np.array(power)[idx_], np.array(energy)[idx_]