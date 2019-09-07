class Memory:

	def __init__(self):
		self.buffer = []

	def add_to_memory(self, cur_state, reward, next_state, end):
		self.buffer.append((cur_state, reward, next_state, end))