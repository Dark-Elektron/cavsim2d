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

```
copy <root folder>/ABCI_MP_12_5/ABCI_MP_12_5/ABCI_MP application for Windows/ABCI_MP_12_5.exe <root folder>/cavsim2d/solver/ABCI
```

Before diving in, I would install `pprintpp`. It is not necessay but it sure does makes the print readable.
```python
import pprint
pp = pprint.PrettyPrinter(indent=4)
```

Examples - TESLA Elliptical Cavity
==================================

There are two fundamental objects in `cavsim2d` - the `Cavities` and `Cavity` objects. A `Cavities` objects holds an array 
of `Cavity` objects while a `Cavity` object contains information about a single RF cavity. They are created as follows:

```python
from cavsim2d.cavity import Cavity, Cavities

cavs = Cavities()
cavs.save(project_folder='/user/home/...')
```

> [!IMPORTANT]
> Cavities().save(<files_path>) must be called first to either create a new project folder or to point to an 
> existing project folder. An extra parameter 'overwrite=True' can be passed to overwrite the project folder if it 
> already exists. Default is 'overwrite=False'.

A `Cavity` object holds information about elliptical cavities. Therefore, a cavity object requires the number of cells,
mid cell, left end cell and right end cell dimensions for its initialisation. We use the 
[TESLA](https://cds.cern.ch/record/429906/files/0003011.pdf) cavity geometry dimensions in this example

```python
# define geometry parameters
n_cells = 9
midcell = [42, 42, 12, 19, 35, 57.7, 103.353]  # <- A, B, a, b, Ri, L, Req
endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]

# create cavity
tesla = Cavity(9, midcell, endcell_l, endcell_r, beampipe='both')
```
The cavity geometry can be viewed using `plot('geometry')` or `cav.inspect()`. `plot('geometry')` returns a 
`matplotlib.axes` object.

```python
tesla.plot('geometry')
# tesla.inspect()
```

Now the cavity can be added to the cavities object.

```python
cavs.add_cavity([tesla], names=['TESLA'], plot_labels=['TESLA'])
```

The `names` keyword is a list of othe names of the Cavities objects. This is the name under which the simulation results
related for the Cavity is saved. The `plot_labels` keyword contain the legend labels. If no entry is made, a default 
name is assigned.

Now we are ready to run our first analysis and print the quantities of interest (qois) for the fundamental mode (FM).

### Eigenmode analysis

```python
cavs.run_eigenmode()
pp.pprint(cavs.eigenmode_qois)
```

Let's try that again but this time using adding a cavity to `cavs`. We will use the a re-entrant cavity geometry. The 
dimensions can be found [here](https://www.sciencedirect.com/science/article/pii/S0168900202016200/pdfft?md5=cb52709f91cc07cfd6e0517e0e6fe49d&pid=1-s2.0-S0168900202016200-main.pdf)
in Table 2. We will use the parameters corresponding to `$\delta e=+30$`. This time we will enter the geometry by defining first a `shape_space`.


```python
shape_space = {'reentrant': 
                   {'IC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],
                    'OC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],
                    'OC_R': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27]
                    }
               }

# create cavity
shape = shape_space['reentrant']
reentrant = Cavity(9, shape['IC'], shape['OC'], shape['OC_R'], beampipe='both')
cavs.add_cavity(reentrant, 'reentrant', 'reentrant')
cavs.plot('geometry')
```

Now we can run the eigenmode simulation once again and print the quantities of interest for the FM.

```python
cavs.run_eigenmode()
pp.pprint(cavs.eigenmode_qois)
```

We can now do is make a comparative bar plot of some FM qois of the two geometries.

```python
cavs.plot_compare_fm_bar()
```

Let's do that again but this time with a single cell without beampipes to compare with [this](https://www.sciencedirect.com/science/article/pii/S0168900202016200/pdfft?md5=cb52709f91cc07cfd6e0517e0e6fe49d&pid=1-s2.0-S0168900202016200-main.pdf).
```python

cavs = Cavities()
cavs.save(project_folder='/user/home/...')

n_cells = 9
midcell = [42, 42, 12, 19, 35, 57.7, 103.353]
tesla_mid_cell = Cavity(1, midcell, midcell, midcell, beampipe='none')

shape_space = {'reentrant': 
                   {'IC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],
                    'OC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],
                    'OC_R': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27]
                    }
               }

# create cavity
shape = shape_space['reentrant']
reentrant_mid_cell = Cavity(1, shape['IC'], shape['IC'], shape['IC'], beampipe='none')

cavs.add_cavity([tesla_mid_cell, reentrant_mid_cell], names=['TESLA', 'reentrant'], plot_labels=['TESLA', 'reentrant'])

ax = cavs.plot('geometry')

cavs.run_eigenmode()
pp.pprint(cavs.eigenmode_qois)

cavs.plot_compare_fm_bar()
```

```python
print(2+3)