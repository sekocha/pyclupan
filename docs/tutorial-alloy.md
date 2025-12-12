# Cluster Expansion Model Estimation for Alloy Systems
In this tutorial, a cluster expansion model will be constructed for the binary Ag–Au and ternary Cu-Ag-Au systems.
The required files for this tutorial can be found in the `examples/Ag-Au` and `examples/Cu-Ag-Au` directories.

In the Ag-Au system, element indices 0 and 1 represent elements Ag and Au, respectively.
In the Cu-Ag-Au system, element indices 0, 1, and 2 represent elements Cu, Ag and Au, respectively.


### 1. Enumeration of Derivative Structures
Starting from the FCC lattice specified by `fcc-primitive`, derivative structures with a given maximum number of atoms are enumerated.
When the maximum number of atoms is given as six, derivative structure enumeration is performed for each number of atoms up to six as follows.
```shell
> for i in {1..6};do
>   pyclupan -p fcc-primitive --supercell_size $i;
>   mv pyclupan_derivatives.yaml pyclupan_derivatives_$i.yaml;
> done
```

For the ternary system, `-e` option can be used to specify ternary elements as follows:
```shell
> pyclupan -p fcc-primitive --supercell_size $i -e 0 1 2;
```

### 2. Derivative Structure Sampling

In this tutorial, the total number of derivative structures is small because the maximum number of atoms is limited to six.
Therefore, POSCAR files for all derivative structures can be generated as follows:
```shell
# Binary
> pyclupan-sample --yaml pyclupan_derivatives_* --method all --element_strings Ag Au
# Ternary
> pyclupan-sample --yaml pyclupan_derivatives_* --method all --element_strings Cu Ag Au
```
The generated derivative structures in POSCAR format will be saved in the `poscars` directory.


### 3. DFT Calculations for Sampled Structures

DFT calculations are performed for the sampled structures.
In this tutorial, we assume that the results of the DFT calculations are already available in the directories `examples/Ag-Au/DFT` and `examples/Cu-Ag-Au/DFT`.


### 4. Nonequivalent Cluster Search

Symmetrycally nonequivalent clusters are enumerated using given cutoff distances.
In this tutorial, we consider clusters up to four-body and their cutoff distances are all 6.0 angstroms.

```shell
# Binary
> pyclupan-cluster -p fcc-primitive -e 0 1 --order 4 --cutoffs 6.0 6.0 6.0
# Ternary
> pyclupan-cluster -p fcc-primitive -e 0 1 2 --order 4 --cutoffs 6.0 6.0 6.0
```

### 5. Estimation of Cluster Expansion Model
#### 5-1. Extract energies from Results of DFT calculations
To prepare a training dataset, energy values from DFT calculations are collected to a file as follows.
```shell
> pyclupan-utils -p fcc-primitive -v DFT/*-*-*/vasprun.xml
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
