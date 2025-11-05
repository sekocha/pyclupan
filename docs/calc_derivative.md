# Enumeration of Derivative Structures
Examples of enumerating derivative structures are shown here.
Derivative structures are defined as nonequivalent substitutional configurations on a given lattice.
These structures will be used to perform DFT calculations for developing cluster expansion models and to identify ground-state structures by calculating the energies of all derivative structures.
The `graphillion` package is additionally required to enumerate the derivative structures.

## Enumeration of Derivative Structures for Single Lattice
### Binary Derivative Structures
Here is an example of enumerating binary derivative structures for a single lattice.
When the expansion degree is set to three, corresponding to possible supercell expansions of size three, all nonequivalent configurations on the lattice are enumerated.
The command line for performing this operation is as follows.
```shell
# > cat poscar-fcc-primitive
# FCC
# 4.0
#   0.000000000  0.5000000000  0.5000000000
#   0.5000000000  0.000000000  0.5000000000
#   0.5000000000  0.5000000000  0.000000000
#  Ag
#   1
# Direct
#   0.000000000 0.000000000 0.000000000

> pyclupan -p poscar-fcc-primitive --supercell_size 3
```
The expansion degree can be given using the `--supercell_size` option.

If the composition of the derivative structures is constrained, the `--comp` option can be used.
In the `--comp` option, the element IDs (starting from zero) and their corresponding compositions must be specified for all elements.
```shell
# --comp element_id chemical_composition
> pyclupan -p fcc-primitive --supercell_size 3 --comp 0 0.3333333333 --comp 1 0.66666666666
```

The compositions do not need to be normalized; equivalently, they can be specified as follows.
```shell
> pyclupan -p fcc-primitive --supercell_size 3 --comp 0 1.0 --comp 1 2.0
```

The composition ranges can also be specified as input parameters.
The lower and upper bounds of the composition for each element can be defined.
```shell
> pyclupan -p fcc-primitive --supercell_size 6 --comp_lb 0 0.5 --comp_ub 0 0.67
```
In this example, only structures in which the composition of element 0 ranges from 0.5 to 0.67 are enumerated.
Note that structures with a composition exactly equal to 0.5 are included.

If a specific supercell matrix is provided instead of the expansion degree, the `--hnf` option can be used.
A given one-dimensional sequence containing nine elements is reshaped into a 3x3 matrix using `np.reshape((3, 3))`.
```shell
# supercell = [[1, 0, 0]
#              [0, 2, 0]
#              [1, 0, 3]]

> pyclupan -p fcc-primitive --hnf 1 0 0 0 2 0 1 0 3
```

After running the derivative structure search, a result file named `derivative.yaml`, containing all possible labelings corresponding to the derivative structures, is generated.

### Beyond Binary Derivative Structures
Here is an example demonstrating the enumeration of ternary derivative structures for a single lattice.
The element types on the lattice are specified using the `-e` option.
In the example below, `-e 0 1 2` indicates that three different elements occupy the same underlying (single) lattice.

```shell
# > cat poscar-fcc-primitive
# FCC
# 4.0
#   0.000000000  0.5000000000  0.5000000000
#   0.5000000000  0.000000000  0.5000000000
#   0.5000000000  0.5000000000  0.000000000
#  Ag
#   1
# Direct
#   0.000000000 0.000000000 0.000000000

(For Ternary)
> pyclupan -p poscar-fcc-primitive --supercell_size 3 -e 0 1 2

(For Quaternary)
> pyclupan -p poscar-fcc-primitive --supercell_size 4 -e 0 1 2 3
```

## Enumeration of Derivative Structures for Multiple Lattices

### Configurations on Single Sublattice

In the following example, binary nonequivalent configurations on the anion sites are enumerated in the cubic perovskite structure.
The elements Sr (index 0) and Ti (index 1) are fixed, and configurations of two elements (indices 2 and 3) are considered on the third sublattice.
The order of the `-e` options is important for controlling the sublattices on which configurations are considered.

```shell
# > cat POSCAR
# Perovskite unit cell
# 1.0
#   4.000000000  0.000000000  0.000000000
#   0.000000000  4.000000000  0.000000000
#   0.000000000  0.000000000  4.000000000
#   Sr Ti O
#   1  1  3
# Direct
#   0.000000000 0.000000000 0.000000000
#   0.5000000000 0.5000000000 0.5000000000
#   0.5000000000 0.5000000000 0.000000000
#   0.000000000 0.5000000000 0.5000000000
#   0.5000000000 0.000000000 0.5000000000

> pyclupan -p POSCAR --supercell_size 4 -e 0 -e 1 -e 2 3
```

The other options `--comp`, `--comp_lb`, `--comp_ub`, and `--hnf` are also available.
To restrict the composition, `--comp_lb` and `--comp_ub` options can be used as follows:
```shell
> pyclupan -p POSCAR --supercell_size 4 -e 0 -e 1 -e 2 3 --comp_lb 2 0.66 --comp_ub 2 0.9
```

### Configurations on Multiple Lattices

In the following example, ternary configurations on the cation A and B sublattices and binary configurations on the anion sublattice are enumerated for the perovskite structure.
Configurations involving three elements (indices 0, 1, and 2) on the first and second sublattices, and two elements (indices 3 and 4) on the third sublattice, are considered.

```shell
# > cat POSCAR
# Perovskite unit cell
# 1.0
#   4.000000000  0.000000000  0.000000000
#   0.000000000  4.000000000  0.000000000
#   0.000000000  0.000000000  4.000000000
#   Sr Ti O
#   1  1  3
# Direct
#   0.000000000 0.000000000 0.000000000
#   0.5000000000 0.5000000000 0.5000000000
#   0.5000000000 0.5000000000 0.000000000
#   0.000000000 0.5000000000 0.5000000000
#   0.5000000000 0.000000000 0.5000000000

> pyclupan -p POSCAR --supercell_size 4 -e 0 1 2 -e 0 1 2 -e 3 4
```

### More Complex Constraints
Here is an example demonstrating how composition constraints are applied.
In this example, ternary configurations on the cation sites and binary configurations on the anion sites are considered for the perovskite structure at the composition of A2BCX5.
Cations A and B occupy both the first and second cation sublattices, while cation C occupies only the second sublattice in the perovskite structure.
Anions X occupy the third sublattice, with one-sixth of this sublattice remaining vacant.

```shell
# > cat POSCAR
# Perovskite unit cell
# 1.0
#   4.000000000  0.000000000  0.000000000
#   0.000000000  4.000000000  0.000000000
#   0.000000000  0.000000000  4.000000000
#   Sr Ti O
#   1  1  3
# Direct
#   0.000000000 0.000000000 0.000000000
#   0.5000000000 0.5000000000 0.5000000000
#   0.5000000000 0.5000000000 0.000000000
#   0.000000000 0.5000000000 0.5000000000
#   0.5000000000 0.000000000 0.5000000000

> pyclupan -p POSCAR --supercell_size 4 -e 0 1 -e 0 1 2 -e 3 4 --comp 0 2.0 --comp 1 1.0 --comp 2 1.0 --comp 3 5.0 --comp 4 1.0
```
Elements 0, 1, and 2 correspond to cations A, B, and C, respectively.
Element 3 corresponds to the anion X, and element 4 represents vacant anion sites.

## Enumeration of Ionic Derivative Structures

The following example enumerates configurations of cations A(2+) and B(3+) on the first sublattice and configurations of cation C(4+) and D(3+) on the second sublattice in perovskite lattice.
Only derivative structures satisfying charge neutrality are enumerated.

```shell
# > cat POSCAR
# Perovskite unit cell
# 1.0
#   4.000000000  0.000000000  0.000000000
#   0.000000000  4.000000000  0.000000000
#   0.000000000  0.000000000  4.000000000
#   Sr Ti O
#   1  1  3
# Direct
#   0.000000000 0.000000000 0.000000000
#   0.5000000000 0.5000000000 0.5000000000
#   0.5000000000 0.5000000000 0.000000000
#   0.000000000 0.5000000000 0.5000000000
#   0.5000000000 0.000000000 0.5000000000

# element 0: A (2+), element 1: B (3+) on sublattice 1
# element 2: C (4+), element 3: D (3+) on sublattice 2
# element 4: X (2-) on sublattice 3

> pyclupan -p POSCAR --supercell_size 4 -e 0 1 -e 2 3 -e 4 --charge 0 2.0 --charge 1 3.0 --charge 2 4.0 --charge 3 3.0 --charge 4 -2.0
```
When enumerating structures that satisfy charge neutrality, the charges of all elements are required.
