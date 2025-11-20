# pyclupan: Cluster expansion tools for alloy and substitutional ionic systems

`pyclupan` is a Python package for developing substitutional cluster expansion models based on datasets obtained from density functional theory (DFT) calculations.
The code also enables the enumeration of nonequivalent substitutional derivative structures, the calculation of correlation (cluster) functions, Monte Carlo simulations, and the evaluation of free energies for multicomponent substitutional systems.

## Citation of pyclupan

"Cluster expansion method for multicomponent systems based on optimal selection of structures for density-functional theory calculations", [A. Seko et al., Phys. Rev. B 80, 165122 (2009)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.165122)

## Required libraries and python modules

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


## How to install pyclupan

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

## How to use pyclupan

- [Enumeration of derivative structures](docs/calc_derivative.md)
