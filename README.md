# pyclupan: Cluster Expansion Tools for Alloy and Substitutional Ionic Systems

`pyclupan` is a Python package for developing substitutional cluster expansion models based on datasets obtained from density functional theory (DFT) calculations.
The code also enables the enumeration of nonequivalent substitutional derivative structures, the calculation of correlation (cluster) functions, Monte Carlo simulations, and the evaluation of free energies for multicomponent substitutional systems.

## Citation of Pyclupan

"Cluster expansion method for multicomponent systems based on optimal selection of structures for density-functional theory calculations", [A. Seko et al., Phys. Rev. B 80, 165122 (2009)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.165122)

## Required Libraries and Python Modules

- python >= 3.9
- numpy != 2.0.*
- scipy
- pyyaml
- setuptools

- pypolymlp >= 0.16.0
- spglib
- phonopy
- openmp (recommended)

[Optional]
- graphillion
- scikit-learn


## How to Install Pyclupan

- Install from PyPI
```
conda create -n pyclupan-env
conda activate pyclupan-env
conda install -c conda-forge numpy scipy spglib
pip install pyclupan
```

- Install from GitHub
```
git clone https://github.com/sekocha/pyclupan.git
cd pyclupan
conda create -n pyclupan-env
conda activate pyclupan-env
conda install -c conda-forge numpy scipy spglib
pip install . -vvv
```

## How to Use Pyclupan
### Tutorials
- [CE Model Estimation for Alloy Systems (Binary Ag-Au and Ternary Cu-Ag-Au)](docs/tutorial-alloy.md)
- [CE Model Estimation for Ionic Substitutional Systems (SrCuO3-x)](docs/tutorial-subs.md)

### Command-Line Interface
- [Enumeration of Derivative Structures](docs/calc_derivative.md)
- [Derivative Structure Sampling](docs/sample_derivative.md)
- [Enumeration of Clusters and Cluster Function Forms](docs/cluster_search.md)
- [Cluster Function Calculations](docs/calc_features.md)
- [Cluster Expansion Model Estimation](docs/regression.md)
- [Energy Calculations](docs/calc_properties.md)
- [Lattice Monte Carlo Simulations](docs/mc.md)
