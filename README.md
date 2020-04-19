### Processing tool for HDF5 dataset result from FLASH code.

The FLASH code is a publicly available high performance application code which has evolved into a modular, 
extensible software system from a collection of unconnected legacy codes.
FLASH includes physics solvers for hydrodynamics, magnetohydrodynamics, nonideal MHD, equation of state, 
radiation transfer, etc.

This tool made for processing results from jet model in CCSN (core-colapse supernova) that was created in FLASH.
Functionality: 
* HDF5 FLASH result file must contain 'ener', 'velx' and 'vely' datasets. 
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

To process calculations of Гb and creating the PDF plot for single HDF5 file or few files, run:  
`python hdf5_reader.py --files [hdf5_file_path,hdf5_file_path,hdf5_file_path]`
Plot will be saved in result_ener_vel_{hdf5_name}.pdf in folder with files.

To find the file with maximum avergage Гb (Lorentz factor*b) from HDF5 files in folder (or list of folders), 
and then process calculations for this file (will be created for each folder), run:  
`python hdf5_reader.py --folders [hdf5_folder_path,hdf5_folder_path,hdf5_folder_path]`
Also will output maximum and average Гb for each file in report.txt.

Additional options:

`--round [0/1/2]`
Set the level to round gammaB value. It will output plot with the level of round, i.e. for round 0 it wt will round all values to digits.
For the higher detalization must be set to 2 (default).

Example:
--round 0       | (4.55 -> 4)
--round 1       | (4.55 -> 4.6)
--round 2       | (4.55 -> 4.55)

`--overlap`
Draw all plots in one graph if there were few files provided. Plots will be labeled by paths of files. Also `overlap` 
will be added to the name of plot file.
