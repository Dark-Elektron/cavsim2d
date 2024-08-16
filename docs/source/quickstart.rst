![GitHub all releases](https://img.shields.io/github/downloads/Dark-Elektron/CavityDesignHub/total?logo=Github)
![GitHub issues](https://img.shields.io/github/issues-raw/Dark-Elektron/CavityDesignHub?logo=Github)
![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/Dark-Elektron/CavityDesignHub?logo=Github)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Dark-Elektron/CavityDesignHub?logo=Github)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/Dark-Elektron/CavityDesignHub?logo=Github)


Quickstart
##########

This repository contains Python codes for designing and analysing 2D axisymmetric RF structures.
The current capabilities include eigenmode analysis, elliptical cavity tuning and optimisation, wakefield analysis,
uncertainty quantification, and quick visualisations and comparisons of results.

Installation
************

To install cadsim2d, clone it into a local directory, `cd` into this directory and run

.. code-block::

    python3 setup.py install


for system-wide installation, or

.. code-block::

    python3 setup.py --user install


for local installation.

Third party code
****************

Wakefield analysis is performed using the ABCI electromagnetic code which solves the Maxwell
equations directly in the time domain when a bunched beam goes through an axisymmetric
structure on or off axis. It is free and can be downloaded from [ABCI](https://abci.kek.jp/abci.htm). Download the latest
version (currently ABCI_MP_12_5.zip). Copy version `ABCI_MP_12_5.exe` from
`<root folder>\ABCI_MP_12_5\ABCI_MP_12_5\ABCI_MP application for Windows` to `<root folder>/cavsim2d/solver/ABCI` or
through the command line with


copy <root folder>/ABCI_MP_12_5/ABCI_MP_12_5/ABCI_MP application for Windows/ABCI_MP_12_5.exe <root folder>/cavsim2d/solver/ABCI


Before diving in, I would install `pprintpp`. It is not necessay but it sure does makes the print readable.

.. code-block::

    import pprint
    pp = pprint.PrettyPrinter(indent=4)


Examples - TESLA Elliptical Cavity
**********************************

The core components of `cavsim2d` are `Cavities` and `Cavity` objects. `Cavities` is a container for multiple `Cavity`
instances, each representing a single RF cavity and its associated data. These objects are instantiated as follows:

.. code-block::
    from cavsim2d.cavity import Cavity, Cavities

    cavs = Cavities()
    cavs.save(project_folder='/user/home/...')

The default name for `Cavities` object is `cavities`. Enter `name` keyword to enter custom name i.e.
`cavs = Cavities('custom_name')`.
This is recommended if you want to run different sets of analysis.


.. tip::

    The location from which you run the program might require adding its directory to the system path using
    `sys.path.append(<cavsim2d_path>)`. For instance, when working from a "notebooks" folder, I typically use:

.. code-block::

    import sys
    sys.path.append("..")


.. important::

    The `Cavities().save(<files_path>)` function initialises or specifies a project folder.
    An optional `overwrite=True` argument can be included to replace an existing folder.
    By default, overwriting is disabled.

A `Cavity` object holds information about elliptical cavities. Therefore, a cavity object requires the number of cells,
mid cell, left end cell and right end cell dimensions for its initialisation. We use the
[TESLA](https://cds.cern.ch/record/429906/files/0003011.pdf) cavity geometry dimensions in this example

.. code-block::
    n_cells = 9
    midcell = [42, 42, 12, 19, 35, 57.7, 103.353]  # <- A, B, a, b, Ri, L, Req
    endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
    endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]

# create cavity
tesla = Cavity(n_cells, midcell, endcell_l, endcell_r, beampipe='both')

The cavity geometry can be viewed using `plot('geometry')` or `cav.inspect()`. All `plot()` functions return a
`matplotlib.axes` object.

.. code-block::

    tesla.plot('geometry')
    # tesla.inspect()


Now the cavity can be added to the cavities object.

.. code-block::

    cavs.add_cavity([tesla], names=['TESLA'], plot_labels=['TESLA'])


The `names` parameter is a list of custom names for each `Cavity` object. These names are used to label
corresponding simulation results. The optional `plot_labels` parameter specifies legend labels for visualizations.
If not provided, default labels will be generated.

Now we are ready to run our first analysis and print the quantities of interest (qois) for the fundamental mode (FM).

Eigenmode analysis
==================

.. code-block::

    cavs.run_eigenmode()
    pp.pprint(cavs.eigenmode_qois)


Let uss try that again but this time using adding a cavity to `cavs`. We will use the a re-entrant cavity geometry. The
dimensions can be found [here](https://www.sciencedirect.com/science/article/pii/S0168900202016200/pdfft?md5=cb52709f91cc07cfd6e0517e0e6fe49d&pid=1-s2.0-S0168900202016200-main.pdf)
in Table 2. We will use the parameters corresponding to `$\delta e=+30$`. This time we will enter the geometry by defining first a `shape_space`.


.. code-block::

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


Now we can run the eigenmode simulation once again and print the quantities of interest for the FM.

.. code-block::

    cavs.run_eigenmode()
    pp.pprint(cavs.eigenmode_qois)


We can now do is make a comparative bar plot of some FM qois of the two geometries.

.. code-block::

    cavs.plot_compare_fm_bar()


Let's do that again but this time with a single cell without beampipes to compare with [this](https://www.sciencedirect.com/science/article/pii/S0168900202016200/pdfft?md5=cb52709f91cc07cfd6e0517e0e6fe49d&pid=1-s2.0-S0168900202016200-main.pdf).

.. code-block::

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


Visualising the mesh and field profiles
=======================================

To visualise the mesh and field profiles use

.. code-block::

    cavs[0].plot_mesh()
    cavs['reentrant'].plot_fields(mode=1, which='E')
    cavs['TESLA'].plot_fields(mode=1, which='H')


.. tip::

    Meshes and fields are properties of a `Cavity` object and not a `Cavities` object. Therefore, to visualise the mesh
    and field profiles, use the `Cavity` object `name` or corresponding index.

Cavity Tuning
=============

Cavity tuning is straightforward using `cavsim2d`. We'll demonstrate this with a TESLA cavity's mid-cell,
initially using an arbitrary equator radius (Req) before converging to the correct value of 103.3 mm.
The tuning function requires at least a tuning parameter and target frequency. For multiple cavities
within a `Cavities` object, these arguments can be provided as lists matching the number of cavities.
Optional parameters can further refine the tuning process.

.. code-block::

    cavs = Cavities()
    cavs.save(project_folder='/user/home/...')

    midcell = [42, 42, 12, 19, 35, 57.7, 100]
    tesla_mid_cell = Cavity(1, midcell, midcell, midcell, beampipe='none')

    cavs.add_cavity(tesla_mid_cell, 'TESLA')
    tune_config = {
        'freqs': 1300,
        'parameters': 'Req',
        'cell_types': 'mid-cell',
        'rerun': True
    }
    cavs.run_tune(tune_config)
    pp.pprint(cavs.eigenmode_tune_res)


.. code-block::

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


Confirm from the output that the correct frequency and `Req` is achieved.

.. note::

    You notice a slight deviation from the 103.353. This is due to the approximation of the mid cell length to 57.7 mm.

Repeat the same calculation. This time retain the correct `Req` and input a wrong `A`.

.. code-block::

    cavs = Cavities()
    cavs.save(project_folder='/user/home/...')

    midcell = [20, 42, 12, 19, 35, 57.7, 103.353]
    tesla_mid_cell = Cavity(1, midcell, midcell, midcell, beampipe='none')

    cavs.add_cavity(tesla_mid_cell, 'TESLA')
    tune_config = {
        'freqs': 1300,
        'parameters': 'A',
        'cell_types': 'mid-cell',
        'processes': 1,
        'rerun': True
    }
    cavs.run_tune(tune_config)
    pp.pprint(cavs.eigenmode_tune_res)


Confirm from the output that the correct frequency and `A` is achieved.


Wakefield
=========

Running wakefield simulations is as easy as running eigenmode simulations described above.

.. code-block::

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


To make plots of the longitudinal and transverse impedance plots on the same axis, we use the following code

.. code-block::

    ax = cavs.plot('ZL')
    ax = cavs.plot('ZT', ax)
    ax.set_yscale('log')


Oftentimes, we want to analyse the loss and kick factors, and higher-order mode power for particular or several
operating points for a cavity geometry. This can easily be done by passing an operating points dictionary to the
`run_wakefield()` function.

.. code-block::

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
    pp.pprint(cavs.wakefield_qois)


And to view the results

.. code-block::

    cavs.plot_compare_hom_bar('Z_SR_4.32mm')

.. important::

    Simulation results are saved in a folder named using the operating point, a specified suffix,
    and the sigma value (format: <operating point name>_<suffix>_<sigma value>mm). To compute higher-order mode
    power, R/Q values are necessary, requiring a prior eigenmode analysis if results are unavailable.


Optimisation
============

Optimisation of cavity geometry can be carried out using cavsim2d. Objective functions that are currently supported
are the fundamental `freq [MHz]`, `Epk/Eacc []`, `Bpk/Eacc [mT/MV/m]`, `R/Q [Ohm]`, `G [Ohm]`, `Q []`, `ZL`, `ZT`.
`ZL` and `ZT` are longitudinal and transverse impedance peaks in specified frequency intervals obtained from wakefield
analysis The algorithm currently implemented is genetic algorithm. The optimisation settings are controlled
using a configuration dictionary. The most important parameters for the algorithm are

- `cell_type`: The options are `mid-cell`, `end-cell` and `end-end-cell` depending on the parameterisation of the cavity
               geometry. See Fig []. Default is `mid-cell`.
  
  'cell_type': 'mid-cell'
  
- `freqs`: Target operating frequency of the cavity.

'parameters': 'Req'

- 'tune freq.': Target operating frequency of the cavity.

`freqs`: 1300


The preceeding parameters belong to the tune_config dictionary and so are entered this way in the optimisation_config

'tune_config': {
    'freqs': 801.58,
    'parameters': 'Req',
    'cell_types': cell_type
}

- `bounds`: This defines the optimisation search space. All geometric variables must be entered.
            Note that variables excluded from optimisation should have identical upper and lower bounds..

'bounds': {'A': [20.0, 80.0],
               'B': [20.0, 80.0],
               'a': [10.0, 60.0],
               'b': [10., 60.0],
               'Ri': [60.0, 85.0],
               'L': [93.5, 93.5],
               'Req': [170.0, 170.0]}


- `objectives`: This defines the objective functions. Objectives could be the minimisation, maximisation of optimisation
             of an objective function to a particular value. They are defined as:

'objectives': [
                ['equal', 'freq [MHz]', 1300],
                ['min', 'Epk/Eacc []'],
                ['min', 'Bpk/Eacc [mT/MV/m]'],
                ['max', 'R/Q [Ohm]'],
                ['min', 'ZL', [1, 2, 5]],
                ['min', 'ZT', [1, 2, 3, 5]]
                ]

The third parameter for the impedances `ZL`, `ZT` define the frequency interval for which to evaluate the peak impedance.
The algorithm specific entries include
- `initial_points`: The number of initial points to be genereated.
- `method`: Method of generating the initial points. Defaults to latin hypercube sampling (LHS).
- `no_of_generations`: The number of generations to be analysed. Defaults to 20.
- `crossover_factor`: The number of crossovers to create offsprings.
- `elites_for_crossover`: The number of elites allowed to produce offsprings.
- `mutation_factor`: The number of mutations to create offsprings.
- `chaos_factor`: The number of new random geometries included to improve diversity.


'initial_points': 5,
'method': {
    'LHS': {'seed': 5},
    },
'no_of_generations': 5,
'crossover_factor': 5,
'elites_for_crossover': 2,
'mutation_factor': 5,
'chaos_factor': 5,

Putting it all together, we get

.. code-block::

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
        'initial_points': 5,
        'method': {
            'LHS': {'seed': 5},
            },
        'no_of_generation': 2,
        'crossover_factor': 5,
        'elites_for_crossover': 2,
        'mutation_factor': 5,
        'chaos_factor': 5
    }

Several other parameters like `method`, can be controlled. The full configuration file can be found in the `config_files` folder.

.. code-block::

    cavs = Cavities()
    # must first save cavities
    cavs.save('/user/home/...')

    cavs.run_optimisation(optimisation_config)


Uncertainty Quantification
===========================

Each simulation described until now can be equiped with uncertainty quantification (UQ) capabilites by passing in a
`uq_config` dictionary. For example, eigenmode F
analysis for a cavity could be carried out including UQ. the same goes for wakefield analysis, tuning, and optimisation.
For example, let's revisit our eigenvalue example.

.. code-block::

    cavs = Cavities()
    cavs.save(project_folder='/user/home/...')

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
        'cell_type': 'mid-cell',
        'cell_complexity': 'simplecell'
    }
    eigenmode_config = {
        'processes': 3,
        'rerun': True,
        'boundary_conditions': 'mm',
        'uq_config': uq_config
    }

    cavs.run_eigenmode(eigenmode_config)
    pp.pprint(cavs.eigenmode_qois)


And to plot the results

.. code-block::

    cavs.plot_compare_fm_bar(uq=True)


.. warning::

    UQ is not yet available for wakefield analysis and cavity tuning.

.. important::

    Enabling uncertainty quantification (UQ) for the original reentrant_mid_cell cavity results in errors due to
    degenerate geometries in its vicinity. Therefore, the `Req` was changed to 110 mm.
    These degeneracies can be identified by using the
    `reentrant_mid_cell.inspect()` to examine and manipulate the cavity's parameters.
    This tool proves invaluable in diagnosing such issues.


Configuration dictionaries
===========================

Simulation inputs are defined through configuration dictionaries, with specific formats for different simulation types.
These dictionaries are structured logically. For instance, a simple eigenmode simulation uses a straightforward
configuration. Uncertainty quantification (UQ) can be integrated by adding a `uq_config` dictionary within the
eigenmode configuration. Wakefield analysis and tuning configurations follow a similar pattern.

Optimisation configurations include a `tune_config` section to ensure frequency optimisation prior to other parameters.
Depending on the optimisation goals, `eigenmode_config` and `wakefield_config` sections can be nested
within the optimisation configuration, potentially also incorporating UQ through `uq_config` sub-dictionaries.

To view the complete configuration dictionaries for each analysis, use the `help()` function,
e.g. `help(cavs.run_eigenmode)`.


.. note::

    Default configuration settings are applied for eigenmode and wakefield analyses when no custom
    configuration dictionary is provided.

Parallelisation
***************

`cavsim2d` simulations can be parallelised easily by setting the `processes` parameter within relevant
configuration dictionaries. This controls the number of processes used for the analysis.
For simulations with uncertainty quantification (UQ) enabled, an additional level of parallelisation can
be achieved by specifying `processes` within the UQ configuration. The default number of processes is one.


Understanding the geometry types
********************************


Folder structure
****************