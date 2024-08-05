import jax
import jax.numpy as jnp
from jax import random
import logging
from typing import Union, Callable, List, Tuple

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class Gene:
    def __init__(self, weight: float, innovation_number: int):
        self.weight = weight
        self.innovation_number = innovation_number


class NEAT(NEAlgorithm):
    """NEAT implementation with historical markings."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 mutation_rate: float = 0.05,
                 mutation_std: float = 0.1,
                 species_threshold: float = 3.0,
                 survival_threshold: float = 0.2,
                 seed: int = 0,
                 logger: logging.Logger = None):
        if logger is None:
            self.logger = create_logger(name='NEAT')
        else:
            self.logger = logger
        self.pop_size = pop_size
        self.param_size = param_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.species_threshold = species_threshold
        self.survival_threshold = survival_threshold
        self.key = random.PRNGKey(seed)

        self.innovation_number = 1
        self.genomes = None
        self.species = []
        self._best_genome = None

    def ask(self) -> jnp.ndarray:
        if self.genomes is None:
            self.key, subkey = random.split(self.key)
            weights = random.normal(subkey, (self.pop_size, self.param_size)) * self.mutation_std
            self.genomes = [[Gene(w, self.get_innovation_number()) for w in genome] for genome in weights]
        return jnp.array([[gene.weight for gene in genome] for genome in self.genomes])

    def tell(self, fitness: Union[jnp.ndarray, jnp.ndarray]) -> None:
        fitness = jnp.array(fitness)

        self.speciate()
        shared_fitness = self.explicit_fitness_sharing(fitness)

        new_population = []
        for species in self.species:
            species_fitness = shared_fitness[species]
            species_genomes = [self.genomes[i] for i in species]

            num_survivors = max(1, int(len(species) * self.survival_threshold))
            survivor_indices = jnp.argsort(species_fitness)[-num_survivors:]
            survivors = [species_genomes[i] for i in survivor_indices]

            num_offspring = len(species) - num_survivors
            offspring = self.generate_offspring(survivors, num_offspring)

            new_population.extend(survivors)
            new_population.extend(offspring)

        self.genomes = new_population
        self._best_genome = self.genomes[jnp.argmax(fitness)]

    def speciate(self):
        self.species = []
        for i, genome in enumerate(self.genomes):
            found_species = False
            for species in self.species:
                if self.genomic_distance(genome, self.genomes[species[0]]) < self.species_threshold:
                    species.append(i)
                    found_species = True
                    break
            if not found_species:
                self.species.append([i])

    def genomic_distance(self, genome1: List[Gene], genome2: List[Gene]) -> float:
        c1, c2, c3 = 1.0, 1.0, 0.4  # Coefficients for balancing distance components

        innovation_numbers1 = set(gene.innovation_number for gene in genome1)
        innovation_numbers2 = set(gene.innovation_number for gene in genome2)

        matching = innovation_numbers1.intersection(innovation_numbers2)
        disjoint = innovation_numbers1.symmetric_difference(innovation_numbers2)

        # Count excess genes
        max_innovation1 = max(innovation_numbers1)
        max_innovation2 = max(innovation_numbers2)
        excess = len([n for n in disjoint if n > max(max_innovation1, max_innovation2)])

        # Calculate average weight difference for matching genes
        weight_diffs = []
        for n in matching:
            gene1 = next(gene for gene in genome1 if gene.innovation_number == n)
            gene2 = next(gene for gene in genome2 if gene.innovation_number == n)
            weight_diffs.append(abs(gene1.weight - gene2.weight))
        avg_weight_diff = sum(weight_diffs) / len(matching) if matching else 0

        N = max(len(genome1), len(genome2))

        return (c1 * excess / N) + (c2 * (len(disjoint) - excess) / N) + c3 * avg_weight_diff

    def explicit_fitness_sharing(self, fitness: jnp.ndarray) -> jnp.ndarray:
        shared_fitness = jnp.zeros_like(fitness)
        for species in self.species:
            species_fitness = fitness[species]
            for i, member in enumerate(species):
                shared_fitness = shared_fitness.at[member].set(
                    species_fitness[i] / len(species)
                )
        return shared_fitness

    def generate_offspring(self, parents: List[List[Gene]], num_offspring: int) -> List[List[Gene]]:
        self.key, *subkeys = random.split(self.key, num_offspring * 3 + 1)
        subkeys = iter(subkeys)

        offspring = []
        for _ in range(num_offspring):
            parent1 = parents[random.randint(next(subkeys), (), 0, len(parents))]
            parent2 = parents[random.randint(next(subkeys), (), 0, len(parents))]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, next(subkeys))
            offspring.append(child)

        return offspring

    def crossover(self, parent1: List[Gene], parent2: List[Gene]) -> List[Gene]:
        child = []
        p1_genes = {gene.innovation_number: gene for gene in parent1}
        p2_genes = {gene.innovation_number: gene for gene in parent2}

        for i in set(p1_genes.keys()) | set(p2_genes.keys()):
            if i in p1_genes and i in p2_genes:
                gene = p1_genes[i] if random.uniform(self.key) < 0.5 else p2_genes[i]
            elif i in p1_genes:
                gene = p1_genes[i]
            else:
                gene = p2_genes[i]
            child.append(Gene(gene.weight, gene.innovation_number))

        return child

    def mutate(self, genome: List[Gene], key: random.PRNGKey) -> List[Gene]:
        mutated_genome = []
        for gene in genome:
            if random.uniform(key) < self.mutation_rate:
                new_weight = gene.weight + random.normal(key) * self.mutation_std
                mutated_genome.append(Gene(new_weight, gene.innovation_number))
            else:
                mutated_genome.append(gene)
        return mutated_genome

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array([gene.weight for gene in self._best_genome])

    @best_params.setter
    def best_params(self, params: Union[jnp.ndarray, jnp.ndarray]) -> None:
        self._best_genome = [Gene(w, self.get_innovation_number()) for w in params]

    def get_innovation_number(self) -> int:
        self.innovation_number += 1
        return self.innovation_number