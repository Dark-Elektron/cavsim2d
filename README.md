![GitHub all releases](https://img.shields.io/github/downloads/Dark-Elektron/CavityDesignHub/total?logo=Github) 
![GitHub issues](https://img.shields.io/github/issues-raw/Dark-Elektron/CavityDesignHub?logo=Github) 
![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/Dark-Elektron/CavityDesignHub?logo=Github) 
![GitHub pull requests](https://img.shields.io/github/issues-pr/Dark-Elektron/CavityDesignHub?logo=Github) 
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/Dark-Elektron/CavityDesignHub?logo=Github)


# Overview

This repository contains Python codes for designing and analysing 2D axisymmetric RF structures. 
The current capabilities include eigenmode analysis, elliptical cavity tuning and optimisation, wakefield analysis, 
uncertainty quantification, and quick visualisations and comparisons of results.

# Installation


To install cadsim2d, clone it into a local directory, `cd` into this directory and run

```python
python3 setup.py install
```

for system-wide installation, or

```python
python3 setup.py --user install
```

for local installation.

### Third party code

Wakefield analysis is performed using the ABCI electromagnetic code which solves the Maxwell
equations directly in the time domain when a bunched beam goes through an axisymmetric
structure on or off axis. It is free and can be downloaded from [ABCI](https://abci.kek.jp/abci.htm). Download the latest
version (currently ABCI_MP_12_5.zip). Copy version `ABCI_MP_12_5.exe` from 
`<root folder>\ABCI_MP_12_5\ABCI_MP_12_5\ABCI_MP application for Windows` to `<root folder>/cavsim2d/solver/ABCI` or 
through the command line with 

```python
copy <root folder>\ABCI_MP_12_5\ABCI_MP_12_5\ABCI_MP application for Windows\ABCI_MP_12_5.exe <root folder>/cavsim2d/solver/ABCI
```

Examples - TESLA Elliptical Cavity
==================================

There are two fundamental objects in cavsim2d - the Cavities and Cavity objects. A Cavities objects holds an array 
of Cavity objects while a Cavity object contains information about a single RF cavity. They are created as follows:

```python
from cavsim2d.cavity import Cavity, Cavities

# create Cavtities object
cavs = Cavities()
# save/point object to simulation directory
cavs.save(files_path='D:\Dropbox\CavityDesignHub\MuCol_Study\SimulationData\ConsoleTest')
```

> [!IMPORTANT]
> Cavities().save(<files_path>) must be called first to either create a new project folder or to point to an 
> existing project folder. An extra parameter 'overwrite=True' can be passed to overwrite the project folder if it 
> already exists. Default is 'overwrite=False'.

A Cavity object holds information about elliptical cavities. Therefore, a cavity object requires the number of cells,
mid cell, left end cell and right end cell dimensions for its initialisation. We use the TESLA cavity geometry 
dimensions in this example

```python
# define geometry parameters
n_cells = 9
midcell = np.array([42, 42, 12, 19, 35, 57.7, 103.353])  # <- A, B, a, b, Ri, L, Req
endcell_l = np.array([40.34, 40.34, 10, 13.5, 39, 55.716, 103.353])
endcell_r = np.array([42, 42, 9, 12.8, 39, 56.815, 103.353])

# create cavity
cav = Cavity(9, midcell, endcell_l, endcell_r, beampipe='both')
```

Now the cavity can be added to the cavities object.

```python
cavs.add_cavity([cav], names=['TESLA'], plot_labels=['TESLA'])
```

The `names` keyword is a list of othe names of the Cavities objects. This is the name under which the simulation results
related for the Cavity is saved. The `plot_labels` keyword contain the legend labels. If no entry is made, a default 
name is assigned. The cavity geometry can be viewed using `plot('geometry')` or `cav.inspect()`.

```python
cav.plot('geometry')
# cav.inspect()
```

Now we are ready to run our first analysis.
### Eigenmode analysis
```python

```
