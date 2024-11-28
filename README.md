# Binless Dynamic Weighted Histogram Analysis Method (DHAM)

## Overview

This repository contains a python implementation of the Binless DHAM method, demonstrated on the simulation of the passage of Na<sup>+</sup> ions through the transmembrane pore of the GLIC channel. This implementation is focused on obtaining a free energy profile. For this example, binless DHAM results are in very good agreement with traditional methods (DHAM, WHAM), using a lagtime of 100 fs and bin number of 1000. Our tests show that the convergence of the profile with respect to bin size and lag time needs to be verified in each case. For DHAM, and Markov state model-based methods in general, smaller bin sizes provide more accurate results, as the diffusion process is closer approximated with better discretization.

This code can be used for analysing data from umbrella sampling simulations. It requires your data to be stored in an `.npy` file. The structure of the data should be such that each row represents a set of data points for a snapshot and each column represents the data across windows. In the example simulation a total of 153 umbrella windows with a uniform spacing of 0.5 Å were employed to determine the free energy.  

## Prerequisites

Before running the scripts, ensure you have  Python 3.x and the following packages:

- NumPy, Matplotlib, Numba

The repository currently includes:

#### Binless_DHAM.py 
The main python script that contains the Binless DHAM implementation for umbrella sampling simulations.

#### coorData.npy
The file storing potential energies from the MD simulation  `coorData.npy`.

#### Clone the repository to your local machine:
```bash
git https://github.com/teodoramateeva/Binless-DHAM
```
#### Make sure you are located in the correct folder: 

```bash
cd Binless-DHAM
```

Make sure the example `.npy` file or the `.npy` file you want to work with exist in the folder.

#### Run:

```bash
python __main__.py
```

You have to modify the `__main__.py` script based on the parameters of your biased simulation.

You might get a warning, which is related to the version of numba, numba==0.56.0 should be installed to supress the warning if needed.

## Citation

For more information and if you use this code in your research, please cite:

https://pubs.acs.org/doi/full/10.1021/acs.jpclett.3c02624
