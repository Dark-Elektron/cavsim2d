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

Examples - TESLA Elliptical Cavity
==================================

There are two fundamental objects in `cavsim2d` - the `Cavities` and `Cavity` objects. A `Cavities` objects holds an array 
of `Cavity` objects while a `Cavity` object contains information about a single RF cavity. They are created as follows:

```python
from cavsim2d.cavity import Cavity, Cavities

cavs = Cavities()
cavs.save(project_folder='D:\Dropbox\CavityDesignHub\MuCol_Study\SimulationData\ConsoleTest')
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
midcell = np.array([42, 42, 12, 19, 35, 57.7, 103.353])  # <- A, B, a, b, Ri, L, Req
endcell_l = np.array([40.34, 40.34, 10, 13.5, 39, 55.716, 103.353])
endcell_r = np.array([42, 42, 9, 12.8, 39, 56.815, 103.353])

# create cavity
tesla = Cavity(9, midcell, endcell_l, endcell_r, beampipe='both')
```

Now the cavity can be added to the cavities object.

```python
cavs.add_cavity([tesla], names=['TESLA'], plot_labels=['TESLA'])
```

The `names` keyword is a list of othe names of the Cavities objects. This is the name under which the simulation results
related for the Cavity is saved. The `plot_labels` keyword contain the legend labels. If no entry is made, a default 
name is assigned. The cavity geometry can be viewed using `plot('geometry')` or `cav.inspect()`.

```python
tesla.plot('geometry')
# cav.inspect()
```

Now we are ready to run our first analysis and print the quantities of interest (qois) for the fundamental mode (FM).
### Eigenmode analysis

```python
cavs.run_eigenmode()
print(cavs.eigenmode_qois)
```

Let's try that again but this time using adding a cavity to `cavs`. We will use the a re-entrant cavity geometry. The 
dimensions can be found [here](https://pdf.sciencedirectassets.com/271580/1-s2.0-S0168900200X05566/1-s2.0-S0168900202016200/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCa4kZrnABWMYzP2OT0j2pIMpx7DRSV7UNkbZKlQSNvnAIhALXmjIb8GDSgO1Pecr89YsYUqSDjREkL6gCVlr5WArLgKrwFCKv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwA8KLH2aOp3npgrDIqkAUIjQ%2F1%2BsX3M2b9h2VANdQlssL7F3b71RSg1yc4DhqtFZcCMpbMyt0QMBTxdok58bVJhknbImkmMWCdc3mQbD9LQjGkNyZRdHytbXEJyf11FoWgDnXxY0Vnmo3e%2FtkCkvWO%2F3fXSuyfwgVmIBi0FaQWxIAybZ363q0MKYksYjktPRnpIu0BxHBFORXQ6gdJeNAq%2BNlJXFS9NYZ8zkzlNqEoQa%2FCesxkKHuYU1WNqNT7h0z1haZj%2FjyV%2F5J2cBqr7DQzdO05e7vknwxmhKGuwGQR5rzzT49KovWQu%2BQ0By2sj4G4drf73tzp84t82p9BdpxgThaPD5TkOsEACBuBbpx7aWK9%2FxrVWAfITYZEl4Y0O5HxG5yCJhEK160mjv5heB2KIVMCNj5IV4ru2WrRlB2HFhviewL8NYTBAiCukKv3u8eJCCreVf7XmJyyH1OzsynbmocjVEI50F3WKMEhUuGL8u4gKDd3mWA3UOugJOcpyh9%2Bn62rEzLoLWOtdZsnT9kZX06Eh3WGrK%2BKAGVGp5L2zKxUCWCco%2FwZKiAKljS933S6uLAV8WAUZ%2Fs%2FIvyv9ts9yu53pgyRCRvUnv5%2FaGtJD%2FKU%2BLKwqBb5Dm1ahkashaqYexjc9oVK%2FpLo0YnKtW2O0LxS21wy6gmVzbNXPo0GScswjlH30ZkRr0oHgf%2B4x2idMAbhsdBiGsc6kBwXyUNc4me8uMU%2FQXlsjLSG5B0%2F1SDTWqHKxn5S8wVYF2pZF5qa8Pd4OkXXTAukweWvQ4RMLnsR8RAvYSE%2Bnm7zcEy1DLV6FW1Vz%2FcfHKt6XhVurHYA07nsu0euiFSbbBMHgSgYkwEP%2BU52F8Y51tAOFFlzhASECev61X077kVOpTKMSTDws%2B61BjqwAZ2%2BNOB0OTqhGCPLRwsZIq1GSZpWEedaWsmPvlkO7B12D6Q9dGwnlLmYJV8acglnEGMIN8ilmTXAnCEoDVAFZHaC8y7q4W%2FKcSGQELoHeaIYDhIpxEf6kRCCttKio%2Fi0G4E8aIleZuNYC1aqrZh51YKoAUwcRAb%2BvKWHYWI76i3Y1bFgzyfVl1fI9v7JUfiBnJtgakj1oLrL8Kk8CvI2wUFW1YGPZItzSRrcsWm9%2FAtE&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240813T181342Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYYYZLDGIR%2F20240813%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=adcaf39a06ba32f054d33d365ed65773099ac4afa7df38ddfee43d8d3ba0a391&hash=3b93597bdeb0e9206afd936159fbada0b7fb05322bfd34371f94389d0593d6f6&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0168900202016200&tid=spdf-68866831-01e2-435c-aa99-271a3287d8b5&sid=0b4db08f3efb774c73984e142713e9bb11c4gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=04005b065004010351&rr=8b2aae3e5bdeca85&cc=de)
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
reentrant = Cavity(2, shape['IC'], shape['OC'], shape['OC_R'], beampipe='both')
cavs.add_cavity(reentrant, 'reentrant', 'reentrant')
cavs.plot('geometry')
```

Now we can run the eigenmode simulation once again and print the quantities of interest for the FM.

```python
cavs.run_eigenmode()
print(cavs.eigenmode_qois)
```

We can now do is make a comparative bar plot of some FM qois of the two geometries.

```python
cavs.plot_compare_fm_bar()
```

