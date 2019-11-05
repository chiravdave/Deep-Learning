import gym

class Pong:

	def __init__(self):
		self.env = gym.make('Pong-v0')
		self.action_mapping = {'up': 2, 'down': 3}

	def reset(self):
		return self.env.reset()

	def show(self):
		self.env.render()

	def close(self):
		self.env.close()
		self.env = None

	def play(self, action):
		next_state, reward, is_done, _ = self.env.step(action)
		return next_state, reward

	def get_all_actions(self):
		return self.action_mapping