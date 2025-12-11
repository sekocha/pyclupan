# Energy Calculations

Using effective cluster interactions obtained from regression, the energies and formation energies for derivative structures can be calculated.
Derivative structures lying on the convex hull of the formation energies are regarded as
stable structures in alloy and substitutional systems.

The command-line interface `pyclupan-calc` with the `--ecis` option allows the calculation of energies, formation energies, and convex hull structures.
The usage of `pyclupan-calc` is similar to that described in [Cluster Function Calculations](calc_features.md).
In addition to the options used for calculating cluster functions, the `--ecis` option must be provided to specify the effective cluster interactions.
When the `--ecis` option is given, the energies and formation energies for the specified
structures and labelings are calculated as follows.

```shell
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --derivatives pyclupan_derivatives_*
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --samples pyclupan_sample_attrs.yaml
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --poscars POSCAR1 POSCAR2 --element_strings Ag Au
```

After executing these commands, three files are generated:
- `pyclupan_energies.hdf5` for energy values,
- `pyclupan_formation_energies.hdf5` for formation energy values, and
- `pyclupan_convexhull.yaml` for the stable structures on the convex hull.

When calculating formation energies, endmembers are defined automatically.
If endmembers are provided manually, the structures in POSCAR format can be specified using the additional `--end_poscars` and `--element_strings` options.

```shell
--end_poscars poscar-end1 poscar-end2 --element_strings Ag Au
```
