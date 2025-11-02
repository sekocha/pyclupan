# pyclupan: Cluster expansion tools for alloy and substitutional ionic systems
`pyclupan` is a Python code 

## Citation of pyclupan

"Cluster expansion method for multicomponent systems based on optimal selection of structures for density-functional theory calculations", [A. Seko et al., Phys. Rev. B 80, 165122 (2009)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.165122)

## Required libraries and python modules

- python >= 3.9
- numpy != 2.0.*
- scipy
- pyyaml
- setuptools
- spglib
- openmp (recommended)


[Optional]
- graphillion


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


