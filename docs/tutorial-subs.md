# Cluster Expansion Model Estimation for Ionic Substitutional Systems
In this tutorial, a cluster expansion model will be constructed for the pseudobinary SrCuO2-SrCuO3 system.
The required files for this tutorial can be found in the `examples/SrCuOx` directory

### 1. Enumeration of Derivative Structures
Starting from the perovskite lattice specified by `perovskite-unitcell`, derivative structures with a given maximum supercell size are enumerated.
When the maximum supercell size is set to four, derivative structure enumeration is performed for supercells containing 3, 6, 9, and 12 sites, as follows.
In addition, the lower bound for the composition is specified corresponding to SrCuO2.

```shell
> for i in {1..4};do
>   pyclupan -p perovskite-unitcell --supercell_size $i -e 0 -e 1 -e 2 3 --comp_lb 2 0.665
>   mv pyclupan_derivatives.yaml pyclupan_derivatives_$i.yaml;
> done
```

### 2. Derivative Structure Sampling
In this tutorial, POSCAR files for all derivative structures are generated as follows:
```shell
> pyclupan-sample --yaml pyclupan_derivatives_* --method all --element_strings Sr Cu O V
```
In the generated POSCAR files, the oxygen vacancy is represented by V.
Therefore, these sites must be removed from the POSCAR files before performing DFT calculations.


### 3. DFT Calculations for Sampled Structures
DFT calculations are performed for the sampled structures.
In this tutorial, we consider that results from DFT calculations are obtained as found in the directory `examples/SrCuOx/DFT`.

### 4. Nonequivalent Cluster Search
Symmetrycally nonequivalent clusters are enumerated using given cutoff distances.
In this tutorial, we consider clusters up to four-body.
The cutoff distance for pairs is set to 10.0 angstroms, while the cutoff distances for all other interactions are set to 6.0 angstroms.

```shell
pyclupan-cluster -p perovskite-unitcell -e 0 -e 1 -e 2 3 --cutoffs 10.0 6.0 6.0
```

### 5. Estimation of Cluster Expansion Model
#### 5-1. Extract energies from Results of DFT calculations
To prepare a training dataset, energy values from DFT calculations are collected to a file as follows.
```shell
> pyclupan-utils -p perovskite-unitcell -v DFT/*-*-*/vasprun.xml
```
File `pyclupan_energy.dat` will be generated.

#### 5-2. Cluster Function Calculations for Sampled Derivative Structures
Cluster functions of the sampled derivative structures are calculated.
```shell
> pyclupan-calc --clusters pyclupan_clusters.yaml --derivatives pyclupan_derivatives_*.yaml
```
An HDF5 File `pyclupan_features.hdf5` will be generated.

#### 5-3. Model Estimation Using Lasso Regression
Using the two files for the training dataset, `pyclupan_energy.dat` and `pyclupan_features.hdf5`, the regression coefficients in the CE model are estimated using Lasso.

```shell
> pyclupan-regression -e pyclupan_energy.dat -f pyclupan_features.hdf5
```
As a result, a file `pyclupan_ecis.yaml` will be generated.
