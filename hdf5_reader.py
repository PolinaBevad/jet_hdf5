import math
import os
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import collections, functools, operator

# Plot axis
RIGHT = 2E2
LEFT = 5E-2
TOP = 1
BOTTOM = 1E-6

FONT_PROPERTIES_TTF_FILE = 'FreeSansBold.ttf'
NXB = 8
NYB = 8

# Unicodes for beautiful symbols of  Lorentz factor and b=v/c
G = u'\u0393'
B = u'\u03B2'


def run_for_one_hdf5(hdf5_file):
    """
    Reads HDF5 file result from hydrodynamic FLASH code.
    Takes density, pressure and velocities,  calculates Lorentz factor
    and create plots of Lorentz factor*b to energy distribution.
    """
    path = os.path.dirname(os.path.abspath(hdf5_file))
    file_name = os.path.basename(hdf5_file)
    file = h5py.File(hdf5_file, 'r')
    print(f"Reports will be saved in folder: {path}")

    # Retrieve all needed datasets
    try:
        velx = file['velx']
        vely = file['vely']
        pres = file['pres']
        dens = file['dens']
    except Exception:
        print('Not specified format, next.')
        return

    # We need values only from indexes with leaf nodes
    # This one contains all nodes, so it can be used for plotting etc.
    node_type_dataset = file['node type']
    indexes, amount_of_indexes = prepare_indexes_at_leaf_node(node_type_dataset)
    round_level = int(args.round)

    # Calculate Lorentz factor: first create empty matrix, then fill with calculated gamma*beta
    gammaB, gamma = prepare_gammaB_dataset(amount_of_indexes)
    gammaB, gamma = calculate_gammaB_dataset(velx, vely, gammaB, gamma, amount_of_indexes, indexes, round_level)

    # Find maximum of gamma*beta for preparation of the plot dictionary
    max_gammaB = check_max_lorentz_factor(gammaB, indexes)

    # Prepare dict for plot
    gamma_dict = prepare_gamma_dict_for_plot(dens, gamma, pres, velx, vely, gammaB, indexes, max_gammaB, round_level)

    total_energy = check_energy_dataset(gamma_dict)
    plot_gammaB_energy(path, file_name, gamma_dict, total_energy)
    return gamma_dict, total_energy


def plot_gammaB_energy(path, file_name, gamma_dict, total_energy):
    """Create the plot of gamma*b to energy in log scale."""
    plot_gamma = list(gamma_dict.keys())
    gamma_array = list(gamma_dict.values())
    plot_ener = []
    for i in range(len(gamma_array)):
        plot_ener.append(sum(gamma_array[i:]))

    plot_ener_integral = [i / total_energy for i in plot_ener]  # ratio of energy to the total energy
    plot_ener_single = [i / total_energy for i in gamma_array]

    prop = FontProperties()  # Font properties to show unicode symbols on plot
    prop.set_file(FONT_PROPERTIES_TTF_FILE)

    name = f'result_ener_vel_{file_name}'
    if not args.overlap:
        plt.clf()
    else:
        name = f'result_ener_vel_overlap_{file_name}'

    # default style for plot
    plt.rcdefaults()
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.subplots_adjust(hspace=0.5)
    plt.style.use('ggplot')
    
    # Plot cumulative
    plt.subplot(2, 1, 1)
    plt.plot(plot_gamma, plot_ener_integral)
    plt.legend()
    plt.xlabel(G + B, fontproperties=prop)
    plt.ylabel(f'E(>{G}{B})/E0', fontproperties=prop)
    plt.ylim(BOTTOM, TOP)
    plt.xlim(LEFT, RIGHT)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Lorentz factor to energy distribution (cumulative)')
    
    # Plot non-cumulative
    plt.subplot(2, 1, 2)
    plt.plot(plot_gamma, plot_ener_single)
    plt.legend()
    plt.xlabel(G + B, fontproperties=prop)
    plt.ylabel(f'E({G}{B})/E0', fontproperties=prop)
    plt.ylim(BOTTOM, TOP)
    plt.xlim(LEFT, 2E2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Lorentz factor to energy distribution (non-cumulative)')
    # Save to file
    plt.savefig(f'{path}\\{name}.pdf', bbox_inches='tight')


def check_energy_dataset(gamma_dict):
    """Calculate summary energy for all prepared values of specified before refine level"""
    total_energy = 0
    for key in gamma_dict:
        total_energy += gamma_dict[key]
    print('Total_energy:', total_energy)
    return total_energy


def prepare_gamma_dict_for_plot(dens, gamma, pres, velx, vely, gammaB, indexes, max_gamma, round_level):
    # Set rounding level for values in dictionary and level of splitting the range of gamma values
    split = 1
    if round_level == 1:
        split = 0.1
    if round_level == 2:
        split = 0.01
    # Fill gamma dict with zero energy
    gammaB_dict = {}
    for i in np.arange(0, max_gamma + 1, split):
        gammaB_dict[round(i, round_level)] = 0
    # For all indexes fill gammaB dictionary
    for i in indexes:
        for k in range(0, NYB):
            for j in range(0, NXB):
                # Add only cells that are not in star, i.e. have velocities
                if velx[i][0][k][j] != 0 or vely[i][0][k][j] != 0:
                    gammaBvalue = gammaB[i][0][k][j]
                    # total energy release for this value of gamma
                    ener = (dens[i][0][k][j] + 4 * pres[i][0][k][j]) * gamma[i][0][k][j] **2 - pres[i][0][k][j]
                    gammaB_dict[gammaBvalue] += ener
        if i % 1000 == 0:
            print(f'Block {i} processed')
    return gammaB_dict


def prepare_gammaB_dataset(len_set):
    """Prepare 3x dimensional datasets that will store data about gamma*b in the cells"""
    gammaB = []
    gamma = []
    for i in range(len_set):
        gammaB.append(0)
        gamma.append(0)
        gammaB[i] = [0]
        gamma[i] = [0]
    print(f"Gamma dataset filled with zeros, length:{len(gammaB)}  and {len(gamma)}.")
    return gammaB, gamma


def calculate_gammaB_dataset(velx_d, vely_d, gammaB_d, gamma_d, len_set, indexes, round_level):
    """Calculate gamma*b value from vel and ener datasets"""
    print('len_set', len_set)
    for i in indexes:
        velx = velx_d[i][0][0:NYB]
        vely = vely_d[i][0][0:NYB]
        B = np.sqrt(velx[0:NXB] ** 2 + vely[0:NXB] ** 2)
        gamma = 1 / np.sqrt(1 - (velx[0:NXB] ** 2 + vely[0:NXB] ** 2))
        gammaB = B * gamma
        gammaB_d[i][0] = np.round(gammaB, round_level)
        gamma_d[i][0] = np.round(gamma, round_level)
    print("Gamma dataset calculated.")
    return gammaB_d, gamma_d


def parse_all_files(folder):
    """Parse all HDF5 files in folder, find the one with the biggest average lorentz factor and plot energy for it."""
    file_gammas = {}
    files = os.listdir(folder)

    for file in files:
        path = f'{folder}\\{file}'
        if os.path.isdir(path):
            print(f'{path} is not file, next.')
            continue
        try:
            hdf5_file = h5py.File(path, 'r')
        except OSError:
            print(f'{path} is not HDF5 file, next.')
            continue

        time = hdf5_file['real scalars'][0][1]  # timestep for this checkpoint

        node_type_dataset = hdf5_file['node type']
        indexes, amount_of_indexes = prepare_indexes_at_leaf_node(node_type_dataset)
        try:
            velx = hdf5_file['velx']
            vely = hdf5_file['vely']
        except ValueError:
            print('Not specified format, next.')
            continue
        max_file_gamma, sum_gamma = calculate_max_gamma(velx, vely, indexes)
        average_gamma = sum_gamma / (len(indexes) * NYB * NXB)
        print(f"Max gamma for file {path} calculated: {max_file_gamma}. Average_gamma: {average_gamma} at time {time}")
        file_gammas[path] = (max_file_gamma, average_gamma, time)

    max_file_path = ''
    max_avg_gamma = 0
    for path, (max_gamma, avg_gamma, time) in file_gammas.items():
        if avg_gamma > max_avg_gamma:
            max_avg_gamma = avg_gamma
            max_file_path = f'{path}'
            
    with open(f'{folder}\\report.txt', mode='w') as output:
        output.write(f'Max avg gamma: {max_avg_gamma} in file: {max_file_path}\n')
        for path, value in file_gammas.items():
            output.write(f'file:{path}\tMax Gamma, Avg Gamma, Timestep:{value}\n')

    run_for_one_hdf5(max_file_path)


def calculate_max_gamma(velx_dataset, vely_dataset, indexes):
    """Calculate max gamma*b and sum of gamma*b value from vel datasets"""
    maxGamma = 0
    sum_gamma = 0
    for i in indexes:
        velx = velx_dataset[i][0]
        vely = vely_dataset[i][0]
        gamma = 1 / np.sqrt(1 - (velx ** 2 + vely ** 2))
        cur_gamma = max(map(max, gamma))
        sum_gamma += sum(map(sum, gamma))
        if cur_gamma > maxGamma:
            maxGamma = cur_gamma
    return maxGamma, sum_gamma


def check_max_lorentz_factor(gammaB_dataset, indexes):
    """Prints and returns the max gamma (lorentz factor) from the dataset.
    Need to find it to determine maximum value for gamma axis in the plot."""
    max_gammaB = 0
    for i in indexes:
        for j in range(0, len(gammaB_dataset[i][0])):
            current_max = max(gammaB_dataset[i][0][j])
            if max_gammaB < current_max:
                max_gammaB = current_max
    print('Max gammaB: ', max_gammaB)
    return max_gammaB


def prepare_indexes_at_leaf_node(node_type):
    """Indexes will be the same in all variable matrixes: ener, velx etc."""
    LEAF_TYPE = 1
    amount_of_indexes = len(node_type[:])
    indexes = []
    for i in range(0, amount_of_indexes):
        if node_type[i] == LEAF_TYPE:
            indexes.append(i)
    return indexes, amount_of_indexes


parser = argparse.ArgumentParser()
parser.add_argument('--files', help="Path to the specific HDF5 files that contains FLASH results. "
                                    "Will generate plots for all of them."
                                    "Can be list of files, split by comma.")
parser.add_argument('--folders', help="Path to the folder with HDF5 files that contains FLASH results. Will parse all "
                                      "files to find the one with the biggest average G and plot it. "
                                      "Also generates table with maximum G for each file on folder. "
                                      "Can be list of folders, split by comma.")
parser.add_argument('--round', help="Round level. If it set to 0, gammaB will be rounded to 1, if it set 1, "
                                    "will be rounded to 0.1, etc.", default=2)
parser.add_argument('--overlap', help="Plot all graphs on one plot (from different studies, for example)",
                    action="store_true")
args = parser.parse_args()


if __name__ == '__main__':
    if args.files:
        files = args.files.split(",")
        for file in files:
            run_for_one_hdf5(file)
    elif args.folders:
        folders = args.folders.split(",")
        for folder in folders:
            parse_all_files(folder)
