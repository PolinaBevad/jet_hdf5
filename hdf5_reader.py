import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

# Path to hdf5 file (checkpoint file)
FONT_PROPERTIES_TTF_FILE = 'FreeSansBold.ttf'
HDF5_FILE_PATH = 'data\dens3.8_pres6.4_ref10_reci4.5_core10_jet1hdf5_chk_0076'

# Unicodes for beautiful symbols of  Lorentz factor and b=v/c
G = u'\u0393'
B = u'\u03B2'


def main():
    """
    Reads HDF5 file result from hydrodynamic FLASH code.
    Takes density, pressure, energy and velocities,  calculates Lorentz factor
    and create plots of Lorentz factor*b to energy distribution.
    """
    # Read HDF5 file with h5py library
    file = h5py.File(HDF5_FILE_PATH, 'r')
    # All comments with ### are for some check or print functions

    ### print_all_available_variables(file)

    # We need values only from indexes with max refine level
    indexes, amount_of_indexes = prepare_indexes_at_max_refine_level(file)

    # Retrieve all needed datasets
    dens_dataset = file['dens']
    pres_dataset = file['pres']
    ener_dataset = file['ener']
    velx_dataset = file['velx']
    vely_dataset = file['vely']

    check_max_density(dens_dataset, indexes)

    # Calculate Lorentz factor: first create empty matrix, then fill with calculated gamma*beta
    gammaB_dataset = prepare_gammaB_dataset(amount_of_indexes)
    calculate_gammaB_dataset(dens_dataset, ener_dataset, velx_dataset, vely_dataset,
                             gammaB_dataset, amount_of_indexes, pres_dataset)

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
    xnew = np.linspace(min(plot_gamma), max(plot_gamma), num=1000)
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

    # For all indexes where needed refinement level is laying, fill gammaB dictionary
    for i in indexes:
        gamma_f = gamma_dataset[i][0][0]
        for j in range(len(gamma_f)):
            gammaBvalue = np.round(gamma_f[j], round_level)
            # sum energy release for this value of gamma
            gamma_dict[gammaBvalue] += ener_dataset[i][0][0][j]
    return gamma_dict


def calculate_gammaB_dataset(dens_dataset, ener_dataset, velx_dataset,
                             vely_dataset, gammaB_dataset, len_set, pres_dataset):
    """
    Calculate gamma*b value from dens, pres, vel and ener datasets
    """
    for i in range(len_set):
        dens = dens_dataset[i][0][0]
        pres = pres_dataset[i][0][0]
        ener = ener_dataset[i][0][0]
        velx = velx_dataset[i][0][0]
        vely = vely_dataset[i][0][0]
        current_gamma = []
        for j in range(len(dens[:])):
            B = math.sqrt(velx[j] ** 2 + vely[j] ** 2)
            gammaB = B * np.sqrt((pres[j] + ener[j]) / (dens[j] + pres[j] * 1.333 / 0.333))
            current_gamma.append(gammaB)
        gammaB_dataset[i][0][0] = current_gamma


def prepare_gammaB_dataset(len_set):
    """
    Prepare 3x dimensional dataset that will store data about gamma*b in the cells
    """
    gammaB_dataset = []
    for i in range(len_set):
        gammaB_dataset.append(0)
        x = [0]
        gammaB_dataset[i] = [0]
        gammaB_dataset[i][0] = [0]
        gammaB_dataset[i][0][0] = [0]
    return gammaB_dataset


def check_max_lorentz_factor(gamma_dataset, indexes):
    """
    Prints and returns the max gamma (lorentz factor) from the dataset.
    Need to find it to determine maximum value for gamma axis in the plot.
    """

    max_gamma = 0
    for i in indexes:
        current_max = max(gamma_dataset[i][0][0])
        if max_gamma < current_max:
            max_gamma = current_max
    print('Max gamma', max_gamma)
    return max_gamma


def prepare_indexes_at_max_refine_level(file):
    """
    Take refine levels that were used for the calculations,
    choose the max one and creates the tables of the indexes of cells that will contain
    values with this maximum refine level.
    Indexes will be the same in all variable matrixes: dens, ener, pres etc.
    """
    # Determine refine level
    refine_level_dataset = file['refine level'][:]
    all_refine_levels = set(refine_level_dataset)
    max_refine_level = max(all_refine_levels)
    print('Max refine level:', max_refine_level)

    # Prepare array of indexes with needed refine level
    amount_of_indexes = len(refine_level_dataset)
    indexes = []
    for i in range(0, amount_of_indexes):
        if refine_level_dataset[i] == max_refine_level:
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
        current_max = max(dens_dataset[i][0][0])
        if max_dens < current_max:
            max_dens = current_max
    print('Max density', max_dens)


if __name__ == '__main__':
    # TODO: add argparser for file
    main()
