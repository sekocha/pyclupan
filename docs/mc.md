# Lattice Monte Carlo Simulations

Using effective cluster interactions obtained from regression, lattice Monte Carlo (MC)
simulations at finite temperatures can be performed.
These MC simulations evaluate the average configurations and average properties at finite temperatures.

The current command-line interface, `pyclupan-mc`, supports canonical and semi-grand
canonical MC simulations.
In addition, simulated annealing for estimating stable configurations is also available.


## Required Files and Parameters

To perform MC simulations, the following files are required:
- The result file from the cluster search, `pyclupan_clusters.yaml`
- The file from regression, `pyclupan_ecis.yaml`.

Other parameters required for performing MC simulations include:
- The supercell size, specified with the `--supercell` option
- The temperatures, specified with the `-t` option
- The numbers of steps per site used for equilibration and averaging, specified with the `--n_steps` option
- The composition for canonical MC, specified with the `--comp` option
- The chemical potentials for semi-grand canonical MC, specified with the `--mu` option

When the number of atoms in the primitive cell differs from that in the conventional
unit cell, the given primitive cell will be refined using `spglib`.
In this case, a supercell is constructed by expanding the refined cell, and the expansion size corresponds to the specified supercell size.
For example, when the FCC primitive cell is provided in `pyclupan_clusters.yaml` and the
supercell size is specified as 3×3×3, the MC simulation cell is composed of a 3×3×3
supercell of the FCC conventional cell, which contains four atoms.

The first and second numbers for the `--n_steps` option are the numbers of steps used for equilibration and averaging, respectively.

## Canonical Monte Carlo Simulation

The composition for canonical MC simulations is specified using the `--comp` option.
The order of the numbers provided with the `--comp` option determines the correspondence
between the compositions and the element indices.
For example, if `--comp 0.2 0.5 0.3` is given, the compositions for elements 0, 1, and 2
are 0.2, 0.5, and 0.3, respectively.

Canonical MC simulations can be initialized using either a random configuration or a user-specified initial configuration.

### Initialization from a Random Configuration

A random configuration is generated based on the composition given by the `--comp` option.

```shell
# Binary
> pyclupan-mc --comp 0.5 0.5 --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
# Ternary
> pyclupan-mc --comp 0.25 0.5 0.25 --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```

### Initialization from a Given Initial Configuration

To provide an initial configuration, a `POSCAR` file and element strings are required to
assign spins corresponding to the elements.
These element strings must be the same as those used when estimating the ECIs.

```shell
> pyclupan-mc -p POSCAR --element_strings Ag Au --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```


## Semi-grand Canonical Monte Carlo simulation

When the `--mu` option is provided, a semi-grand canonical MC simulation is performed.
The chemical potential differences, measured from the first active element, are specified
with this option.
For example, if `--mu 0.2 -0.2` is given, the chemical potential differences between
elements 1 and 0, and between elements 2 and 0, are 0.2 and –0.2, respectively.


### Initialization from a Random Configuration

A random configuration is generated based on the composition given by the `--comp` option.

```shell
# Binary
> pyclupan-mc --mu 0.5 --comp 0.5 0.5 --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
# Ternary
> pyclupan-mc --mu 0.25 0.1 --comp 0.25 0.5 0.25 --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```

### Initialization from a Given Initial Configuration

To provide an initial configuration, a `POSCAR` file and element strings are required to
assign spins corresponding to the elements.
These element strings must be the same as those used when estimating the ECIs.

```shell
> pyclupan-mc --mu 0.5 -p POSCAR --element_strings Ag Au --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```

## Simulated Annealing for Evaluating the Ground-State Structure

To perform simulated annealing for estimating the ground-state structure at a fixed
composition, the `--simulated_annealing` option can be used.
In simulated annealing, three temperature parameters must be provided:
`--temp_init`, `--temp_final`, and `--n_temps`.
The temperatures are generated using `np.logspace(temp_init, temp_final, n_temps)`.

Simulated annealings can also be initialized using either a random configuration or a user-specified initial configuration.

```shell
> pyclupan-mc --simulated_annealing --comp 0.5 0.5 --clusters pyclupan_clusters.yaml --ecis pyclupan_ecis.yaml --supercell 2 2 2 --n_steps 100 1000 --temp_init 1000 --temp_final 10 --n_temps 10
```
