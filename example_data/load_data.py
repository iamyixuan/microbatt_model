import glob
import numpy as np
import pandas as pd

def load_example_data(dir_path):
    test_data_dir = dir_path
    files = glob.glob(test_data_dir +'*.csv')
    
    #print("Loading file %s" % files)
    currents = []
    cell_pot = []
    capacity = []
    energy = []
    power = []
    for f in files:
        #print("Loading file %s" % f)
        temp = pd.read_csv(f)
        currents.append(temp['Curr den (A/m^2)'].values[0])
        cell_pot.append(temp['Cell pot(V)'].values)
        capacity.append(temp['Capacity'].values)
        energy.append(temp['Energy'].values[0])
        power.append(temp['Power'].values[0])

    idx_ = np.argsort(currents)
    return [cell_pot[i] for i in idx_], [capacity[i] for i in idx_], np.array(currents)[idx_], np.array(power)[idx_], np.array(energy)[idx_]

