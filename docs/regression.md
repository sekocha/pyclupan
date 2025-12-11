# Cluster Expansion Model Estimation

Effective cluster interactions are estimated from DFT calculations for derivative structures and their cluster functions using linear regression techniques.
The cluster expansion model, defined by the effective cluster interactions, can be used to calculate formation energies for derivative structures and to perform Monte Carlo
simulations at finite temperatures.

## Preparation of Training Dataset

Before estimating effective cluster interactions using regression, a training dataset must be prepared.
The training dataset consists of cluster functions and energy values for derivative structures.
Cluster functions can be calculated as demonstrated in [Cluster Function Calculations](calc_features.md).
Energy data can be extracted from DFT calculations when using VASP as follows:

```shell
> pyclupan-utils -p fcc-primitive -v $path_dft/*/vasprun.xml
```
By collecting the DFT data, a `pyclupan_energy.dat` file is generated.
The unit cell structure is used to evaluate the energy per unit cell.
The generated `pyclupan_energy.dat` file contains the energy values in eV per unit cell.


## Model Estimation from Training Dataset

Effective cluster interactions can then be estimated from the training dataset as follows:

```shell
> pyclupan-regression -e pyclupan_energy.dat -f pyclupan_features.hdf5
```
The Lasso technique is used to estimate the interactions.
Once the regression is completed, a file named `pyclupan_ecis.yaml` containing the effective cluster interactions will be generated.
