import random

from MCSSAlgos.Domains.DomainBase import Domain


def create_initial_sets(domain: Domain, initial_set_sizes: list):
    elements = domain.get_elements()
    initial_sets = []
    for size in initial_set_sizes:
        new_set = set(random.sample(elements, size))
        initial_sets.append(new_set)
        elements.difference_update(new_set)

    return initial_sets