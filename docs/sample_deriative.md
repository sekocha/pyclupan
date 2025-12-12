# Derivative Structure Sampling

The number of derivative structures can often become extremely large, so structure files are not generated during the enumeration process.
To obtain structure files for derivative structures, you can use the `pyclupan-sample` command-line interface.


When sampling all derivative structures in the labeling representation included in the YAML files, use the `--method all` option:

```shell
pyclupan-sample --yaml pyclupan_derivatives_* --method all --element_strings Ag Au
```
The `--element_strings` option is required to assign actual elements to the element labels used in the derivative structures.
In this example, element labels 0 and 1 are assigned to Ag and Au, respectively.
The generated structure files are saved in the poscar directory along with a summary of the sampled structures.

To sample structures uniformly across different supercell shapes, use the `--method uniform option` together with `-n` to specify the number of structures to sample:

```shell
pyclupan-sample --yaml pyclupan_derivatives_* --method uniform -n 10 --element_strings Ag Au
```

To sample structures randomly from the set of derivative structures, use the `--method random` option with `-n`:
```shell
pyclupan-sample --yaml pyclupan_derivatives_* --method random -n 10 --element_strings Ag Au
```
