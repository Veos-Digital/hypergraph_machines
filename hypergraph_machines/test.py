import torch
from hypergraph_machines.meta_learning.HypergraphMachine import Space, Morphism


def test_pruned_morphism():
    m = Morphism(1,2,3, prunable = False)
    n = Morphism(1,2,3, prunable = True)
    tol = 100
    m.prune(tol)
    n.prune(tol)
    print(m.pruned, n.pruned)

def test_pruned_space():
    incoming = [Morphism(1,2,3), Morphism(1,2,3)]
    [setattr(m, "_pruned", True) for m in incoming]
    s = Space((1,1), (2,2), 3, incoming_morphisms = incoming)
    if s.pruned == True:
        print("space pruning passed")
    else:
        raise ValueError("space pruning failed")


if __name__ == "__main__":
    test_pruned_space()
    test_pruned_morphism()
