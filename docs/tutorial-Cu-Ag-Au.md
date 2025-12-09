# Development of Cluster Expansion Model in the Ternary Cu-Ag-Au System

```shell
> pyclupan-cluster -p fcc-primitive -e 0 1 2 --order 4 --cutoffs 6.0 6.0 6.0
> pyclupan-utils -p ../1-enum/fcc-primitive -v ../3-dft/*-*-*/vasprun.xml
> pyclupan-calc --clusters pyclupan_clusters.yaml --derivatives ../1-enum/pyclupan_derivatives_*
> pyclupan-regression -e energy.dat -f pyclupan_features.hdf5
```
