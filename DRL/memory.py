from numpy.random import choice

class ReplayBuffer:

	def __init__(self):
		self.buffer = []

	def add_to_memory(self, cur_state, action, next_state, reward):
		self.buffer.append((cur_state, action, next_state, reward))

	def batch(self, batch_size):
		batch_indexes = choice(len(self.buffer), size=batch_size, replace=False)
		batch = [self.buffer[index] for index in batch_indexes]
		return batch

	def clear(self):
		self.buffer = []