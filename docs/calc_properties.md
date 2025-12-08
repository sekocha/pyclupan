# Energy Calculations

## Energy and formation energies for binary derivative structures
In addition to the options used for calculating correlation functions, the `--ecis` option is needed to provide effective cluster interactions.
When `--ecis` option is given, the energies and formation energies for given structures and labelings are calculated.
```shell
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --derivatives pyclupan_derivatives_*
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --samples pyclupan_sample_attrs.yaml
> pyclupan-calc --ecis pyclupan_ecis.yaml --clusters pyclupan_cluster.yaml --poscars derivative-1 derivative-1 --element_strings Ag Au
```
