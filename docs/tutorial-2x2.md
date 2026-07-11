# Cluster Expansion Model Estimation for Substitutional Systems on Multiple Sublattices
In this tutorial, we will construct a cluster expansion model for the pseudobinary SiC-AlN system.
Elements Si and Al (labeled 0 and 1) are assigned to the first sublattice, while elements C and N (labeled 2 and 3) are assigned to the second sublattice.
The files required for this tutorial can be found in the `examples/SiC-AlN` directory.

### 1. Enumeration of Derivative Structures
Starting from the wurtzite lattice specified by `wurtzite-primitive`, derivative structures with a given maximum supercell size are enumerated.
When the maximum supercell size is set to four, derivative structure enumeration is performed for supercells containing 4, 8, 12, and 16 sites, as follows.

```shell
> for i in {1..4};do
>   pyclupan-derivatives -p wurtzite-primitive --supercell_size $i -e 0 1 -e 2 3 --charge 0 4.0 --charge 1 3.0 --charge 2 -4.0 --charge 3 -3.0
>   mv pyclupan_derivatives.yaml pyclupan_derivatives_$i.yaml;
> done
```
To enumerate derivative structures in the pseudobinary SiC-AlN system, the charge values for Si, Al, C, and N are set to 4, 3, -4, and -3, respectively.
Only structures with a total charge of zero will be enumerated.

### 2. Derivative Structure Sampling
In this tutorial, POSCAR files for all derivative structures are generated as follows:
```shell
> pyclupan-sample --yaml pyclupan_derivatives_* --method uniform -n 500 --element_strings Si Al C N
```
Derivative structures are sampled uniformly across all possible supercells.

### 3. DFT Calculations for Sampled Structures
DFT calculations are performed for the sampled structures.
In this tutorial, we consider that results from DFT calculations are obtained as found in the directory `examples/SiC-AlN/DFT`.

### 4. Nonequivalent Cluster Search
Symmetrycally nonequivalent clusters are enumerated using given cutoff distances.
In this tutorial, we consider clusters up to four-body.
The cutoff distance for all interactions are set to 6.0 angstroms.

```shell
pyclupan-cluster -p wurtzite-primitive -e 0 1 -e 2 3 --cutoffs 6.0 6.0 6.0
```

### 5. Estimation of Cluster Expansion Model
#### 5-1. Extract energies from Results of DFT calculations
To prepare a training dataset, energy values from DFT calculations are collected to a file as follows.
```shell
> pyclupan-utils -p wurtzite-primitive -v DFT/*-*-*/vasprun.xml
```
File `pyclupan_energy.dat` will be generated.

#### 5-2. Cluster Function Calculations for Sampled Derivative Structures
Cluster functions of the sampled derivative structures are calculated.
```shell
> pyclupan-features --clusters pyclupan_clusters.yaml --derivatives pyclupan_derivatives_*.yaml
```
An HDF5 File `pyclupan_features.hdf5` will be generated.

#### 5-3. Model Estimation Using Lasso Regression
Using the two files for the training dataset, `pyclupan_energy.dat` and `pyclupan_features.hdf5`, the regression coefficients in the CE model are estimated using Lasso.

```shell
> pyclupan-regression -e pyclupan_energy.dat -f pyclupan_features.hdf5
```
As a result, files `pyclupan_ecis.yaml` will be generated.
