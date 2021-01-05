from numpy.random import choice
from numpy import array, ndarray, float32, int32
from typing import Tuple

class ReplayBuffer:

	def __init__(self):
		self.states = []
		self.v_values = []
		self.actions_taken = []
		self.rewards = []
		self.are_terminals = []

	def add_to_memory(self, state: ndarray, v_value: float, action_taken: int, reward: float, is_terminal: bool) -> None:
		"""
		This method will save the given transition into the memory buffer.
		"""

		self.states.append(state)
		self.v_values.append(v_value)
		self.actions_taken.append(action_taken)
		self.rewards.append(reward)
		self.are_terminals.append(is_terminal)

	def batch(self, batch_size: int, correct_q_values: ndarray) -> Tuple[ndarray, ndarray]:
		"""
		This method will return a batch of transitions randomly selected from the memory buffer.
		"""

		batch_indexes = choice(len(self.rewards)-1, size=batch_size, replace=False)
		batch_states = array([self.states[index] for index in batch_indexes], dtype=float32)
		batch_actions_taken = array([self.actions_taken[index] for index in batch_indexes], dtype=int32)
		batch_correct_q_values = correct_q_values[batch_indexes]
		return batch_states, batch_actions_taken, batch_correct_q_values

	def clear(self) -> None:
		"""
		This method will clear the memory buffer.
		"""

		self.states.clear()
		self.v_values.clear()
		self.actions_taken.clear()
		self.rewards.clear()
		self.are_terminals.clear()
