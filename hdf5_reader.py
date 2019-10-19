import math
import os
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

# Path to hdf5 file (checkpoint file)
FONT_PROPERTIES_TTF_FILE = 'FreeSansBold.ttf'
INTERPOLATION = 1000
NXB = 8
NYB = 8

# Unicodes for beautiful symbols of  Lorentz factor and b=v/c
G = u'\u0393'
B = u'\u03B2'


def main(hdf5_file):
    """
    Reads HDF5 file result from hydrodynamic FLASH code.
    Takes density, pressure, energy and velocities,  calculates Lorentz factor
    and create plots of Lorentz factor*b to energy distribution.
    All comments with ### are for some check or print functions
    """
    # Read HDF5 file with h5py library
    file = h5py.File(hdf5_file, 'r')

    print_all_available_variables(file)

    # This one contains all nodes, so it can be used for plotting etc.
    node_type_dataset = file['node type']

    # We need values only from indexes with leaf nodes
    indexes, amount_of_indexes = prepare_indexes_at_leaf_node(node_type_dataset)

    # Retrieve all needed datasets
    dens_dataset = file['dens']
    pres_dataset = file['pres']
    ener_dataset = file['ener']
    velx_dataset = file['velx']
    vely_dataset = file['vely']

    ### check_max_density(dens_dataset, indexes)

    # Calculate Lorentz factor: first create empty matrix, then fill with calculated gamma*beta
    gammaB_dataset = prepare_gammaB_dataset(amount_of_indexes)
    calculate_gammaB_dataset(dens_dataset, ener_dataset, velx_dataset, node_type_dataset, vely_dataset,
                             gammaB_dataset, amount_of_indexes, pres_dataset, hdf5_file, indexes)

    # Find maximum of gamma*beta for preparation of the plot dictionary
    max_gammaB = check_max_lorentz_factor(gammaB_dataset, indexes)

    # Prepare dict for plot
    round_level = 0
    gamma_dict = prepare_gamma_dict_for_plot(ener_dataset, gammaB_dataset, indexes, max_gammaB, round_level)

    total_energy = check_energy_dataset(gamma_dict)
    # Draw plot
    plot_gammaB_ener(gamma_dict, total_energy)


def only_plot_gamma(gamma_file):
    """
    Reads HDF5 file with gamma (already calculated) and
    create plots of Lorentz factor*b to energy distribution.
    """
    gamma_file = h5py.File(gamma_file, 'r')
    # We need values only from indexes with max refine level
    refine_level_dataset = gamma_file["refine level"]
    indexes, amount_of_indexes = prepare_indexes_at_leaf_node(refine_level_dataset)

    gammaB_dataset = gamma_file["gamma"]
    ener_dataset = gamma_file["ener"]

    # Find maximum of gamma*beta for preparation of the plot dictionary
    max_gammaB = check_max_lorentz_factor(gammaB_dataset, indexes)

    # Prepare dict for plot
    round_level = 0
    gamma_dict = prepare_gamma_dict_for_plot(ener_dataset, gammaB_dataset, indexes, max_gammaB, round_level)

    total_energy = check_energy_dataset(gamma_dict)
    # Draw plot
    plot_gammaB_ener(gamma_dict, total_energy)


def plot_gammaB_ener(gamma_dict, total_energy):
    """
    Create the plot of gamma*b to energy in log scale.
    """
    plot_gamma = list(gamma_dict.keys())
    # ratio of energy to the total energy
    plot_ener = [i / total_energy for i in list(gamma_dict.values())]

    # Font properties to show unicode symbols on plot
    prop = FontProperties()
    prop.set_file(FONT_PROPERTIES_TTF_FILE)

    # default style for plot
    plt.rcdefaults()

    # Interpolate a little for smoothness
    f = interp1d(plot_gamma, plot_ener, kind='cubic')
    xnew = np.linspace(min(plot_gamma), max(plot_gamma), num=INTERPOLATION)
    plt.plot(xnew, f(xnew))

    # Labels and logarithmic scale
    plt.xlabel(G + B, fontproperties=prop)
    plt.ylabel('E({}{})/E0'.format(G, B), fontproperties=prop)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Lorentz factor to energy distribution')

    plt.show()


def print_all_available_variables(file):
    """
    Prints dict of all available properties and variables in HDF5 file
    """
    for key in file.keys():
        print(key)


def check_energy_dataset(gamma_dict):
    """
    Calculates summary energy for all prepared values of specified before refine level
    """
    total_energy = 0
    for key in gamma_dict:
        total_energy += gamma_dict[key]
    print('Total_energy:', total_energy)
    return total_energy


def prepare_gamma_dict_for_plot(ener_dataset, gamma_dataset, indexes, max_gamma, round_level):
    # Set rounding level for values in dictionary and level of splitting the range of gamma values
    split = 1
    if round_level == 0:
        split = 1
    if round_level == 1:
        split = 0.1

    # Fill gamma dict with zero energy
    gamma_dict = {}
    for i in np.arange(0, max_gamma + 1, split):
        gamma_dict[round(i, round_level)] = 0
    # For all indexes fill gammaB dictionary
    for i in indexes:
        for k in range(0, NYB):
            gamma_f = gamma_dataset[i][0][k]
            for j in range(0, NXB):
                gammaBvalue = np.round(gamma_f[j], round_level)
                # sum energy release for this value of gamma
                gamma_dict[gammaBvalue] += ener_dataset[i][0][k][j]
    print("Gamma plot prepared.")
    return gamma_dict


def calculate_gammaB_dataset(dens_dataset, ener_dataset, velx_dataset, node_type_dataset,
                             vely_dataset, gammaB_dataset, len_set, pres_dataset, hdf5_file, indexes):
    """
    Calculate gamma*b value from dens, pres, vel and ener datasets
    """
    print('len_set', len_set)
    for i in indexes:
        for k in range(0, NYB):
            dens = dens_dataset[i][0][k]
            pres = pres_dataset[i][0][k]
            ener = ener_dataset[i][0][k]
            velx = velx_dataset[i][0][k]
            vely = vely_dataset[i][0][k]
            current_gamma = []
            for j in range(0, NXB):
                B = math.sqrt(velx[j] ** 2 + vely[j] ** 2)
                gammaB = B * np.sqrt((pres[j] + ener[j]) / (dens[j] + pres[j] * 1.333 / 0.333))
                current_gamma.append(gammaB)
            gammaB_dataset[i][0][k] = current_gamma
    print("Gamma dataset calculated.")

    # Save the gamma dataset, energy and refinement levels to file so it can be used later\
    with h5py.File('gammaB_' + os.path.basename(hdf5_file), 'w') as f:
        f.create_dataset_like("gamma", dens_dataset)
        f.create_dataset_like("ener", ener_dataset)
        f.create_dataset_like("node type", node_type_dataset)
        f['gamma'][...] = gammaB_dataset
        f['ener'][...] = ener_dataset
        f['node type'][...] = node_type_dataset


def prepare_gammaB_dataset(len_set):
    """
    Prepare 3x dimensional dataset that will store data about gamma*b in the cells
    """
    gammaB_dataset = []
    for i in range(len_set):
        gammaB_dataset.append(0)
        gammaB_dataset[i] = [0]
        gammaB_dataset[i][0] = list([0 for j in range(0, NYB)])
    print("Gamma dataset filled with zeros.")
    return gammaB_dataset


def check_max_lorentz_factor(gamma_dataset, indexes):
    """
    Prints and returns the max gamma (lorentz factor) from the dataset.
    Need to find it to determine maximum value for gamma axis in the plot.
    """
    max_gamma = 0
    for i in indexes:
        for j in range(0, len(gamma_dataset[i][0])):
            current_max = max(gamma_dataset[i][0][j])
            if max_gamma < current_max:
                max_gamma = current_max
    print('Max gamma: ', max_gamma)
    return max_gamma


def prepare_indexes_at_leaf_node(node_type):
    """
    Indexes will be the same in all variable matrixes: dens, ener, pres etc.
    """
    LEAF_TYPE = 1
    amount_of_indexes = len(node_type[:])
    indexes = []
    for i in range(0, amount_of_indexes):
        if node_type[i] == LEAF_TYPE:
            indexes.append(i)
    return indexes, amount_of_indexes


def check_max_density(dens_dataset, indexes):
    """
    Calculates and prints maximum density (to check it with VisIt). Uses only indexes
    with maximum refine level
    """
    # Max density
    max_dens = 0
    for i in indexes:
        for j in range(0, len(dens_dataset[i][0])):
            current_max = max(dens_dataset[i][0][j])
            if max_dens < current_max:
                max_dens = current_max
    print('Max density: ', max_dens)


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f', "--HDF5", help="Path to the HDF5 file that contains FLASH results. ")
group.add_argument('-g', "--gamma", help="Path to gamma dataset. Will process only plots preparation and drawing.")
parser.add_argument('-i', "--inter", help="Number of intervals for the cubic interpolation of result. Default 1000.",
                    required=False)

if __name__ == '__main__':
    # TODO: add multithreading?
    args = parser.parse_args()
    if args.inter:
        INTERPOLATION = args.inter

    if args.gamma:
        only_plot_gamma(args.gamma)
    else:
        main(args.HDF5)
