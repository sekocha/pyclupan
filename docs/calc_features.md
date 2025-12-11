# Cluster Function Calculations

Cluster function values can be calculated using symmetrically independent clusters, specified by the `--clusters` option.
The calculated cluster functions will be saved to an HDF5 file `pyclupan_features.hdf5`.

When cluster functions are computed for enumerated derivative structures, the `--derivatives` option can be used to specify the files containing the enumerated labelings of those derivative structures.

```shell
> pyclupan-calc --clusters pyclupan_cluster.yaml --derivatives pyclupan_derivatives_*.yaml
```

When cluster functions are calculated for sampled structures taken from a set of derivative structures, use the `--sample` option to specify the sampled structures.

```shell
> pyclupan-calc --clusters pyclupan_cluster.yaml --samples pyclupan_sample_attrs.yaml
```

If structures in POSCAR format are provided, cluster functions can be calculated as follows.
```shell
> pyclupan-calc --clusters pyclupan_cluster.yaml --poscars POSCAR1 POSCAR2 --element_strings Ag Au
```
In this case, the element strings must be specified to define their corresponding element indices.
The order of the element strings is important, as it determines the element indices.
In this example, the elements Ag and Au are assigned element indices 0 and 1, respectively.
