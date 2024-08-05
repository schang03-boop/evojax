import jax
import jax.numpy as jnp
from jax import random
import logging
from typing import Union, Callable

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class NEAT(NEAlgorithm):
    """A simple NEAT (NeuroEvolution of Augmenting Topologies) implementation."""

    def __init__(self,
                 param_size: int,
                 pop_size: int,
                 init_stdev: float = 0.1,
                 mutation_rate: float = 0.05,
                 mutation_std: float = 0.1,
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
        self.key = random.PRNGKey(seed)

        self.params = None
        self._best_params = None

        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)

    def ask(self) -> jnp.ndarray:
        if self.params is None:
            self.key, subkey = random.split(self.key)
            self.params = random.normal(subkey, (self.pop_size, self.param_size)) * self.mutation_std
        return self.jnp_stack(self.params)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        fitness = jnp.array(fitness)
        elite_indices = jnp.argsort(fitness)[-self.pop_size // 2:]
        elite_params = self.params[elite_indices]

        self.key, *subkeys = random.split(self.key, 4)
        subkey1, subkey2, subkey3 = subkeys

        def create_offspring(_, keys):
            parent1 = elite_params[random.randint(keys[0], (), 0, elite_params.shape[0])]
            parent2 = elite_params[random.randint(keys[1], (), 0, elite_params.shape[0])]
            child = self.crossover(parent1, parent2, keys[2])
            return self.mutate(child, keys[3])

        keys = random.split(subkey1, (self.pop_size - elite_params.shape[0], 4))
        offspring = jax.vmap(create_offspring)(jnp.arange(self.pop_size - elite_params.shape[0]), keys)

        self.params = jnp.concatenate([elite_params, offspring])
        self._best_params = self.params[jnp.argmax(fitness)]

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._best_params = jnp.array(params)

    def crossover(self, parent1: jnp.ndarray, parent2: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        mask = random.uniform(key, parent1.shape) < 0.5
        return jnp.where(mask, parent1, parent2)

    def mutate(self, params: jnp.ndarray, key: random.PRNGKey) -> jnp.ndarray:
        mask_key, noise_key = random.split(key)
        mask = random.uniform(mask_key, params.shape) < self.mutation_rate
        noise = random.normal(noise_key, params.shape) * self.mutation_std
        return jnp.where(mask, params + noise, params)