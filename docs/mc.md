# Lattice Monte Carlo Simulations

## Canonical Monte Carlo simulation
### From a random configuration
```shell
> pyclupan-mc --comp 0.5 0.5 --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```
When the numbers of atoms in primitive cells and the conventional unitcell are different, the given primitive cell will be refined using `spglib`.
In this situation, a supercell is constructed by the expansion of the refined cell and its expansion size corresponds to the given supercell size.


### From a given initial configuration
```shell
> pyclupan-mc -p POSCAR_init --element_strings Ag Au --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500
```
To provide an initial configuration, `POSCAR` file and element strings are required for specifying spins corresponding to elements.
This element strings must be the same as used to estimate ECIs.


## Semi-grand Canonical Monte Carlo simulation
To perform a semi-grand canonical MC simulation, chemical potential values are provided as follows. If `--mu` option is activated, the simulation will be automatically performed in semi-grand canonical ensemble.
A randomly-generated initial configuration is provided by the composition given by the `--comp` option.
### From a random configuration
```shell
> pyclupan-mc --comp 0.5 0.5 --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500 --mu 0.5
```

### From a given initial configuration
```shell
> pyclupan-mc -p POSCAR_init --element_strings Ag Au --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 3 3 3 --n_steps 100 1000 -t 500 --mu 0.5
```

## Simulated annealing for evaluaing ground state structure
To perform a simulated annealing for estimating the ground state state at a fixed composition, `--simulated_annealing` option can be used.
In the simulated annealing, three temperature parameters of `--temp_init`, `-temp_final`, and `--n_temps` must be given.
The temperatures will be set by `np.logspace(temp_init, temp_final, n_temps)`.

### From a random configuration
```shell
> pyclupan-mc --simulated_annealing --comp 0.5 0.5 --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 2 2 2 --n_steps 100 1000 --temp_init 1000 --temp_final 10 --n_temps 10
```

### From a given initial configuration
```shell
> pyclupan-mc --simulated_annealing -p POSCAR_init --element_strings Ag Au --clusters ./pyclupan_clusters.yaml --ecis ./pyclupan_ecis.yaml --supercell 2 2 2 --n_steps 100 1000 --temp_init 1000 --temp_final 10 --n_temps 10
```
