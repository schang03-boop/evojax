import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Sequence, Tuple, Dict, Any
import json

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


class NEATCompatibleMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str
    output_activation: str

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.get_activation(self.activation)(x)
        x = nn.Dense(self.output_dim)(x)
        return self.get_activation(self.output_activation)(x)

    @staticmethod
    def get_activation(name):
        if name == 'relu':
            return nn.relu
        elif name == 'tanh':
            return nn.tanh
        elif name == 'sigmoid':
            return nn.sigmoid
        elif name == 'softmax':
            return lambda x: nn.softmax(x, axis=-1)
        else:
            return lambda x: x  # pass-through


class NEATCompatibleMLPPolicy(PolicyNetwork):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Sequence[int] = (20, 20),
                 activation: str = 'tanh',
                 output_activation: str = 'tanh',
                 logger=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        self.logger = logger or create_logger(name='NEATCompatibleMLPPolicy')

        self.model = NEATCompatibleMLP(
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_activation=output_activation
        )

        # Initialize parameters
        key = random.PRNGKey(0)
        dummy_input = jnp.ones((1, input_dim))
        self.params = self.model.init(key, dummy_input)

        # Prepare jitted functions
        self.forward = jax.jit(self.model.apply)
        self.get_actions_jit = jax.jit(self.get_actions)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        return self.forward(params, t_states.obs), p_states

    def mutate_params(self, params: Dict[str, Any], mutation_rate: float, mutation_std: float, key: random.PRNGKey) -> \
    Dict[str, Any]:
        """Mutate the parameters for NEAT compatibility.

        This method applies mutations to the neural network parameters to maintain
        compatibility with the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

        Args:
            params (Dict[str, Any]): The parameters to mutate.
            mutation_rate (float): The probability of mutating each parameter.
            mutation_std (float): The standard deviation of the mutation noise.
            key (random.PRNGKey): The random key for generating mutations.

        Returns:
            Dict[str, Any]: The mutated parameters.
        """
        def mutate(param, subkey):
            mask_key, noise_key = random.split(subkey)
            mask = random.uniform(mask_key, param.shape) < mutation_rate
            noise = random.normal(noise_key, param.shape) * mutation_std
            return jnp.where(mask, param + noise, param)

        keys = random.split(key, jax.tree_util.tree_leaves(params).__len__())
        return jax.tree_map(mutate, params, keys)

    def crossover_params(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two sets of parameters for NEAT compatibility."""
        key = random.PRNGKey(0)

        def crossover(p1, p2):
            mask = random.uniform(key, p1.shape) < 0.5
            return jnp.where(mask, p1, p2)

        return jax.tree_map(crossover, params1, params2)

    def save_model(self, filename: str):
        """Save the model parameters to a file."""
        with open(filename, 'w') as f:
            json.dump(jax.tree_map(lambda x: x.tolist(), self.params), f)

    def load_model(self, filename: str):
        """Load the model parameters from a file."""
        with open(filename, 'r') as f:
            loaded_params = json.load(f)
        self.params = jax.tree_map(lambda x: jnp.array(x), loaded_params)

    @property
    def num_params(self):
        return sum(p.size for p in jax.tree_leaves(self.params))