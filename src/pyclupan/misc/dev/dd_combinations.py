#!/usr/bin/env python
import numpy as np
from graphillion import GraphSet


class DDCombinations:

    def __init__(self, components, weight=None):

        self.components = components
        if weight is None:
            self.weight = np.ones(len(components))
        else:
            self.weight = weight

        self.nodes = [(int(c), int(c), w) for c, w in zip(self.components, self.weight)]

    def sum_weight(self, lb=-np.inf, ub=np.inf, tol=1e-10, return_combinations=True):

        universe_original = None
        if len(GraphSet.universe()) != 0:
            universe_original = GraphSet.universe()

        GraphSet.set_universe(self.nodes)
        lconst = [(self.nodes, (lb - tol, ub + tol))]
        gs = GraphSet.graphs(linear_constraints=lconst)

        if return_combinations:
            combs = self.to_combinations(gs)

        if universe_original is not None:
            GraphSet.set_universe(universe_original)

        if return_combinations:
            return combs
        return gs

    def to_combinations(self, gs):
        return [[g2[0] for g2 in g1] for g1 in gs]


if __name__ == "__main__":

    n = 5
    components = list(range(n))
    weight = [1] * len(components)
    #    weight = None

    comb_obj = DDCombinations(components, weight=weight)
    combs = comb_obj.sum_weight(lb=4.0)
    #    combs = comb_obj.to_combinations(gs)
    print(combs)
