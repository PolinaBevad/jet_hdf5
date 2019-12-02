### Processing tool for HDF5 dataset result from FLASH code.

The FLASH code is a publicly available high performance application code which has evolved into a modular, 
extensible software system from a collection of unconnected legacy codes.
FLASH includes physics solvers for hydrodynamics, magnetohydrodynamics, nonideal MHD, equation of state, 
radiation transfer, etc.

This tool made for processing results from jet model in CCSN (core-colapse supernova) that was created in FLASH.
Functionality: 
* HDF5 FLASH result file must contain 'dens', 'ener', 'pres', 'velx' and 'vely' datasets. 
* Can take HDF5 file, calculate Г*b values.
* Creates plot for E(Гb) (energy distribution per Гb).

**Г - Lorentz factor**  
**b = v/c**

### Requirements
* Python 3.6
* h5py
* Numpy
* Matplotlib

Versions of required libraries are in *requirements.txt*

### Usage

To process calculations of Гb and creating the plot, run:  
`python hdf5_reader.py -f [hdf5_file_path]`

Option `-t` can be used to calculate integral energy. Otherwise energy will be output only for the current lorentz factor.


