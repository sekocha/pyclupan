# Development of Cluster Expansion Model in the Binary Ag-Au System
In this tutorial, a cluster expansion model will be constructed for the binary Ag–Au
system.
The required files for this tutorial can be found in the `examples/Ag-Au` directory.

### 1. Enumeration of Derivative Structures
Starting from the FCC lattice specified by `fcc-primitive`, derivative structures with a given maximum number of atoms are enumerated.
When the maximum number of atoms is given as six, derivative structure enumeration is performed for each number of atoms up to six as follows.
```shell
> for i in {1..6};do
>   pyclupan -p fcc-primitive --supercell_size $i;
>   mv pyclupan_derivatives.yaml pyclupan_derivatives_$i.yaml;
> done
```

### 2. Derivative Structure Sampling
Coming soon.
- Generation of POSCAR files used for DFT calculations.

### 3. DFT Calculations for Sampled Structures
DFT calculations are performed for the sampled structures.
In this tutorial, we consider that results from DFT calculations are obtained as found in the directory `examples/Ag-Au/DFT`.

### 4. Nonequivalent Cluster Search
Symmetrycally nonequivalent clusters are enumerated using given cutoff distances.
In this tutorial, we consider clusters up to four-body and their cutoff distances are all 6.0 angstroms.

```shell
> pyclupan-cluster -p fcc-primitive -e 0 1 --order 4 --cutoffs 6.0 6.0 6.0
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
