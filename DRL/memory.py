from numpy.random import choice

class ReplayBuffer:

	def __init__(self):
		self.buffer = []

	def add_to_memory(self, cur_state, action, next_state, reward):
		"""
		This method will save the given transition into the memory buffer.

		:param cur_state: current game state
		:param action: action performed in the current game state
		:param next_state: next game state
		:param reward: reward earned after performing the action
		"""

		self.buffer.append((cur_state, action, next_state, reward))

	def batch(self, batch_size):
		"""
		This method will return a batch of transitions randomly selected from the memory buffer.

		:param batch_size: size of the batch to be returned
		:rtype: batch of randomly selected transitions
		"""

		batch_indexes = choice(len(self.buffer), size=batch_size, replace=False)
		batch = [self.buffer[index] for index in batch_indexes]
		return batch

	def clear(self):
		"""
		This method will clear the memory buffer.
		"""

		self.buffer = []