# Special Quasirandom Structure (SQS) Generation

In this tutorial, a special quasirandom structure (SQS) is constructed for alloy and substitutional systems.
An SQS mimics a disordered structure for a given composition by minimizing the difference between the ideal cluster functions and those of the SQS for a specified set of clusters.
The following approach is based on simulated annealing, which searches for an SQS within a given supercell.

The required files for this tutorial can be found in the `examples/Ag-Au` and `examples/Cu-Ag-Au` directories.
In the Ag–Au system, element indices 0 and 1 represent Ag and Au, respectively.
In the Cu–Ag–Au system, element indices 0, 1, and 2 represent Cu, Ag, and Au, respectively.

### 1. Nonequivalent Cluster Search

First, symmetrically nonequivalent clusters are enumerated using specified cutoff distances.
The enumerated clusters define the SQS as specified by the user.
In this tutorial, we consider only pair clusters whose cluster functions are compatible with the ideal cluster functions.
Moreover, the cutoff distance for pair clusters is set to 6.0 angstroms.

```shell
# Binary
> pyclupan-cluster -p fcc-primitive -e 0 1 --order 2 --cutoffs 6.0

# Ternary
> pyclupan-cluster -p fcc-primitive -e 0 1 2 --order 2 --cutoffs 6.0
```
File `pyclupan_clusters.yaml` will be generated after these commands are successfully executed.

### 2. SQS Search

Using the selected set of clusters, an SQS is searched for using simulated annealing.
The input parameters include the composition of the SQS, the supercell size, three parameters defining the initial temperature, final temperature, and number of temperature steps, as well as the number of Monte Carlo steps.

When the number of atoms in the primitive cell differs from that in the conventional unit cell, the given primitive cell is refined using `spglib`.
In this case, a supercell is constructed by expanding the refined cell, and the expansion size corresponds to the specified supercell size.

```shell
# Binary
> pyclupan-sqs --comp 0.25 0.75 --element_strings Ag Au --clusters pyclupan_clusters.yaml --supercell 3 3 3 --n_steps 100 --temp_init 100 --temp_final 1 --n_temps 20

# Ternary
> pyclupan-sqs --comp 0.25 0.25 0.5 --element_strings Cu Ag Au --clusters pyclupan_clusters.yaml --supercell 3 3 3 --n_steps 100 --temp_init 100 --temp_final 1 --n_temps 20
```
As a result, an SQS is generated and written to the file `POSCAR-SQS`.
