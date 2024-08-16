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

--------------------------------------------
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
tesla = Cavity(n_cells, midcell, endcell_l, endcell_r, beampipe='both')
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

## Eigenmode analysis

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
reentrant = Cavity(n_cells, shape['IC'], shape['OC'], shape['OC_R'], beampipe='both')
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

cavs.add_cavity([tesla_mid_cell, reentrant_mid_cell], 
                names=['TESLA', 'reentrant'], 
                plot_labels=['TESLA', 'reentrant'])

ax = cavs.plot('geometry')

cavs.run_eigenmode()
pp.pprint(cavs.eigenmode_qois)

cavs.plot_compare_fm_bar()
```

## Visualising the mesh and field profiles

To visualise the mesh and field profiles use

```python
cavs[0].plot_mesh()
cavs['reentrant'].plot_fields(mode=1, which='E')
cavs['TESLA'].plot_fields(mode=1, which='H')
```

> [!TIP]
> Meshes and fields are properties of a Cavity object and not a Cavities object. Therefore, to visualise the mesh
> and field profiles, use the `Cavity` object `name` or corresponding index.

---------------

## Configuration dictionaries

Configuration dictionaries are used to specify simulation inputs. Each simulation has its specific configuration dictonary
format and content. The configuration files are written so that it is logical. For example, a simple eigenmode simulation
config file is shown below:


We will talk about uncertainty quantification (UQ) later but if we were to equip the eigenmode analysis with UQ,
we only need include a uq_config dictioanry as an entry into the eigenmode_config dictionary. For example:


The same goes for wakefield analysis and tuning. For optimisation control, consider that in an optimisation, we
want to optimise for a particular frequency so for any parameter set generated, we want to first tunr. Therefore,
an optimisation_config dictionary will contain a tune_config. The depending on the objectives the optimisation_config
then includes an eigenmode_config and or a wakefield_config field. It also follows that the eigenmode_config and 
wakefield_config can also contain uq_config fields.

> [!NOTE]
> For eigenmode and wakefield analysis, if a configuration dictionary is not entered, default values are used.

---------------
## Cavity Tuning

Cavity tuning can easily be done using `cavsim2d`. Let's start from the mid cell of a TESLA cavity geometry. We know 
that the correct eqator radius `Req` equals 103.3. However, we start from an arbitrary `Req` to demonstrate tuning.
Two arguments are required - the parameter to tune and the target frequency. Other parameters are optional. Since 
`Cavities` contains several `Cavity` objects, the parameter and frequency arguments can also be a list with length 
corresponding to the lengths of the len of the `Cavities` object.
 
```
TESLA
{   'TESLA': {   'CELL TYPE': 'mid cell',
                 'FREQ': 1300.0007857768796,
                 'IC': [   42.0,
                           42.0,
                           12.0,
                           19.0,
                           35.0,
                           57.7,
                           103.3702896505612, # <- Req
                           103.27068613930538],
                 'OC': [   42.0,
                           42.0,
                           12.0,
                           19.0,
                           35.0,
                           57.7,
                           103.3702896505612,
                           103.27068613930538],
                 'OC_R': [   42.0,
                             42.0,
                             12.0,
                             19.0,
                             35.0,
                             57.7,
                             103.3702896505612,
                             103.27068613930538],
                 'TUNED VARIABLE': 'Req'}}
```

Confirm from the output that the correct frequency and `Req` is achieved. 
> [!NOTE]
> You notice a slight deviation from the 103.353. This is due to the approximation of the mid cell length to 57.7 mm.

Repeat the same calculation. This time retain the correct `Req` and input a wrong `A`.

```python
cavs = Cavities()
cavs.save(project_folder='/user/home/...')

midcell = [20, 42, 12, 19, 35, 57.7, 103.353]
tesla_mid_cell = Cavity(1, midcell, midcell, midcell, beampipe='none')

cavs.add_cavity(tesla_mid_cell, 'TESLA')

cavs.run_tune('A', 1300)
pp.pprint(cavs.eigenmode_tune_res)
```

Confirm from the output that the correct frequency and `A` is achieved.

-------------------------------------------
## Wakefield

Running wakefield simulations is as easy as running eigenmode simulations described above. 

```python
from cavsim2d.cavity import Cavity, Cavities
import pprint
pp = pprint.PrettyPrinter(indent=4)

cavs = Cavities()
cavs.save(project_folder='/user/home/...')

# define geometry parameters
n_cells = 9
midcell = [42, 42, 12, 19, 35, 57.7, 103.353]  # <- A, B, a, b, Ri, L, Req
endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]

# create cavity
tesla = Cavity(n_cells, midcell, endcell_l,endcell_r, beampipe='none')
cavs.add_cavity([tesla], names=['TESLA'], plot_labels=['TESLA'])

cavs.run_wakefield()
```
To make plots of the longitudinal and transverse impedance plots on the same axis, we use the following code

```python
ax = cavs.plot('ZL')
ax = cavs.plot('ZT', ax)
ax.set_yscale('log')
```

In the example, we passed a single parameter `bunch_length`. However, the simulation would have run without this 
argument using internal default arguments for the wakefield solver. 

Oftentimes, we want to analyse the loss and kick factors, and higher-order mode power for particular or several 
operating points for a cavity geometry. This can easily be done by passing an operating points dictionary to the 
`run_wakefield()` function.

```python
op_points = {
            "Z": {
                "freq [MHz]": 400.79,  # Operating frequency
                "E [GeV]": 45.6,  # <- Beam energy
                "I0 [mA]": 1280,  # <- Beam current
                "V [GV]": 0.12,  # <- Total voltage
                "Eacc [MV/m]": 5.72,  # <- Accelerating field
                "nu_s []": 0.0370,  # <- Synchrotron oscillation tune
                "alpha_p [1e-5]": 2.85,  # <- Momentum compaction factor
                "tau_z [ms]": 354.91,  # <- Longitudinal damping time
                "tau_xy [ms]": 709.82,  # <- Transverse damping time
                "f_rev [kHz]": 3.07,  # <- Revolution frequency
                "beta_xy [m]": 56,  # <- Beta function
                "N_c []": 56,  # <- Number of cavities
                "T [K]": 4.5,  # <- Operating tempereature
                "sigma_SR [mm]": 4.32,  # <- Bunch length
                "sigma_BS [mm]": 15.2,  # <- Bunch length
                "Nb [1e11]": 2.76  # <- Bunch population
            }
}
wakefield_config = {
    'bunch_length': 25,
    'wakelength': 50,
    'processes': 2,
    'rerun': True,
    'operating_points': op_points,
}
cavs.run_wakefield(wakefield_config)
pp.pprint(cavs.abci_qois)
```

And to view the results

```python
cavs.plot_compare_hom_bar('Z_SR_4.32mm')
```
> [!IMPORTANT]
> Simulation results are saved in a folder named using the operating point, a specified suffix, 
> and the sigma value (format: <operating point name>_<suffix>_<sigma value>mm). To compute higher-order mode 
> power, R/Q values are necessary, requiring a prior eigenmode analysis if results are unavailable.


------------------------------------------------------

## Optimisation

Optimisation of cavity geometry can be carried out using cavsim2d. Objective functions that are currently supported 
are the fundamental `freq [MHz]`, `Epk/Eacc []`, `Bpk/Eacc [mT/MV/m]`, `R/Q [Ohm]`, `G [Ohm]`, `Q []`, `ZL`, `ZT`. 
`ZL` and `ZT` are longitudinal and transverse impedance peaks in specified frequency intervals obtained from wakefield
analysis The algorithm currently implemented is genetic algorithm. The optimisation settings are controlled 
using a configuration dictionary. The most important parameters for the algorithm are 

- `cell_type`: The options are `mid-cell`, `end-cell` and `end-end-cell` depending on the parameterisation of the cavity
               geometry. See Fig []. Default is `mid-cell`.
  ```
  'cell_type': 'mid-cell'
  ```
- `freqs`: Target operating frequency of the cavity.
```
'parameters': 'Req'
```
- 'tune freq.': Target operating frequency of the cavity.
```
`freqs`: 1300
```

The preceeding parameters belong to the tune_config dictionary and so are entered this way in the optimisation_config
```
'tune_config': {
    'freqs': 801.58,
    'parameters': 'Req',
    'cell_types': cell_type
}
```
- `bounds`: This defines the optimisation search space. All geometric variables must be entered. 
            Note that variables excluded from optimisation should have identical upper and lower bounds..
```
'bounds': {'A': [20.0, 80.0],
               'B': [20.0, 80.0],
               'a': [10.0, 60.0],
               'b': [10., 60.0],
               'Ri': [60.0, 85.0],
               'L': [93.5, 93.5],
               'Req': [170.0, 170.0]}
```

- `objectives`: This defines the objective functions. Objectives could be the minimisation, maximisation of optimisation
             of an objective function to a particular value. They are defined as:
```
'objectives': [
                ['equal', 'freq [MHz]', 1300],
                ['min', 'Epk/Eacc []'],
                ['min', 'Bpk/Eacc [mT/MV/m]'],
                ['max', 'R/Q [Ohm]'],
                ['min', 'ZL', [1, 2, 5]],
                ['min', 'ZT', [1, 2, 3, 5]]
                ]
```
The third parameter for the impedances `ZL`, `ZT` define the frequency interval for which to evaluate the peak impedance.
The algorithm specific entries include
- `initial_points`: The number of initial points to be genereated.
- `method`: Method of generating the initial points. Defaults to latin hypercube sampling (LHS).
- `no_of_generations`: The number of generations to be analysed. Defaults to 20.
- `crossover_factor`: The number of crossovers to create offsprings.
- `elites_for_crossover`: The number of elites allowed to produce offsprings.
- `mutation_factor`: The number of mutations to create offsprings.
- `chaos_factor`: The number of new random geometries included to improve diversity.

```
'initial_points': 5,
'method': {
    'LHS': {'seed': 5},
    },
'no_of_generations': 5,
'crossover_factor': 5,
'elites_for_crossover': 2,
'mutation_factor': 5,
'chaos_factor': 5,
```
Putting it all together, we get
```python
optimisation_config = {
    'tune_config': {
        'freqs': 1300,
        'parameters': 'Req',
        'cell_types': 'mid-cell',
        'processes': 1
    },
    'bounds': {'A': [20.0, 80.0],
               'B': [20.0, 80.0],
               'a': [10.0, 60.0],
               'b': [10., 60.0],
               'Ri': [60.0, 85.0],
               'L': [93.5, 93.5],
               'Req': [170.0, 170.0]},
    'objectives': [
        # ['equal', 'freq [MHz]', 801.58],
                      ['min', 'Epk/Eacc []'],
                      ['min', 'Bpk/Eacc [mT/MV/m]'],
                      # ['min', 'ZL', [1, 2, 5]],
                  ],
    'initial points': 5,
    'method': {
        'LHS': {'seed': 5},
        },
    'no. of generation': 2,
    'crossover factor': 5,
    'elites for crossover': 2,
    'mutation factor': 5,
    'chaos factor': 5
}
```
Several other parameters like `method`, can be controlled. The full configuration file can be found in the `config_files` folder.

```python
cavs = Cavities()
# must first save cavities
cavs.save('D:\Dropbox\CavityDesignHub\MuCol_Study\SimulationData\ConsoleTest')

cavs.run_optimisation(optimisation_config)
```

## Uncertainty Quantification Capabilities

Each simulation described until now can be equiped with uncertainty quantification (UQ) capabilites by passing in a
`uq_config` dictionary. For example, eigenmode 
analysis for a cavity could be carried out including UQ. the same goes for wakefield analysis, tuning, and optimisation.
For example, let's revisit our eigenvalue example.

```python
cavs = Cavities()
cavs.save(project_folder='D:\Dropbox\CavityDesignHub\MuCol_Study\SimulationData\ConsoleTest')

midcell = [42, 42, 12, 19, 35, 57.7, 103.353]
tesla_mid_cell = Cavity(1, midcell, midcell, midcell, beampipe='none')

shape_space = {'reentrant': 
                   {'IC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 110],
                    'OC': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 110],
                    'OC_R': [53.58, 36.58, 8.08, 9.84, 35, 57.7, 110]
                    }
               }

# create cavity
shape = shape_space['reentrant']
reentrant_mid_cell = Cavity(1, shape['IC'], shape['IC'], shape['IC'], beampipe='none')

cavs.add_cavity([tesla_mid_cell, reentrant_mid_cell], 
                names=['TESLA', 'reentrant'], 
                plot_labels=['TESLA', 'reentrant'])

uq_config = {
    'option': True,
    'variables': ['L', 'Req'],
    'objectives': ["freq [MHz]", "R/Q [Ohm]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "G [Ohm]", "kcc [%]", "ff [%]"],
    'delta': [0.05, 0.05],
    'method': ['Quadrature', 'Stroud3'],
    'cell type': 'mid-cell',
    'cell complexity': 'simplecell'
}
eigenmode_config = {
    'processes': 3,
    'rerun': True,
    'boundary_conditions': 'mm',
    'uq_config': uq_config
}

cavs.run_eigenmode(uq_config=uq_config)
pp.pprint(cavs.eigenmode_qois)

```

And to plot the results

```python
cavs.plot_compare_fm_bar(uq=True)
```

> [!IMPORTANT]
> UQ is not yet available for wakefield analysis and cavity tuning.

> [!IMPORTANT]
> Enabling uncertainty quantification (UQ) for the original reentrant_mid_cell cavity results in errors due to 
> degenerate geometries in its vicinity. Therefore, the `Req` was changed to 110 mm. 
> These degeneracies can be identified by using the 
> `reentrant_mid_cell.inspect()` to examine and manipulate the cavity's parameters. 
> This tool proves invaluable in diagnosing such issues.


## Parallelisation

Simulations using `cavsim2d` are easily parallelised by specifying a value for the argument `processes`
specifying the amount of processes the analysis should use. If UQ is enabled, an extra level of parallelisation 
can be achieved by passing `processes` also in the uq configuration dictionary. The number of processes defaults to 1.

## Understanding the geometry types



## Folder structure