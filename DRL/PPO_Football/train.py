import gfootball.env as football_env
import numpy as np
import 

# Gloabl vars & hyper-params
LAMBDA = 0.95
GAMMA = 0.997
LEARNING_RATE = 1e-4
CLIP = 0.115
CRITIC_LOSS_WEIGHT = 0.5
ITERS = int(2e7) # 20M
EPOCHS = 2
BATCH = 4
TARGET_SCORE = 5
TIMESTEPS = 100 # TODO: configurable parameter

def cal_advantages(rewards, values, terminals):
	gae = 0.0
	all_gae = []
	n_samples = len(rewards)

	for i in range(n_samples-1, -1, -1):
	# An episode ends at a terminal state. Hence, gae will be 0.
		if terminals[i]:
			gae = 0.0
		else:
			delta = rewards[i] + GAMMA * values[i+1] - values[i]
			gae = delta + GAMMA * LAMBDA * gae
		all_gae.insert(0, gae)

	advantages = np.array(all_gae)

	return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

def cal_correct_V_values(rewards, values, terminals):
	correct_V_values = []
	n_samples = len(rewards)

	for i in range(n_samples):
	"""
	An episode ends at a terminal state. We don't want terminal state to affect training hence
	making correct value same as predicted value.
	"""
	if terminals[i]:
		correct_V_values.append(values[i])
	else:
		V_value = rewards[i] + GAMMA * values[i+1]
		correct_V_values.append(values[i])

	return np.array(correct_V_values)

def actor_loss(self, cur_policy_probs, old_policy_probs, advantages):
	policy_diff = np.log(cur_policy_probs) - np.log(old_policy_probs)
	cpi_values = policy_diff * advantages
	clipped_cpi_values = tf.clip_by_value(policy_diff, 1 - CLIP, 1 + CLIP, name="clipped_cpi_values") * advantages

	return tf.math.reduce_mean(tf.math.minimum(cpi_values, clipped_cpi_values) * -1, name="actor_loss")

def critic_loss(self, predicted_values, correct_values):
	squared_diff = tf.math.squared_difference(predicted_values, correct_values)
	
	return tf.math.reduce_mean(squared_diff, name="critic_loss")

def ppo_loss(self, predicted_values, correct_values, cur_policy_probs, old_policy_probs, advantages):
	critic_loss = critic_loss(predicted_values, correct_values)
	actor_loss = actor_loss(cur_policy_probs, old_policy_probs, advantages)
	
	return actor_loss + CRITIC_LOSS_WEIGHT * critic_loss

def train():
	actions = ["action_idle", "action_left", "action_top_left", "action_top", "action_top_right", "action_right", 
	"action_bottom_right", "action_bottom", "action_bottom_left", "action_long_pass", "action_high_pass", 
	"action_short_pass", "action_shot", "action_sprint", "action_release_direction", "action_release_sprint", 
	"action_sliding", "action_dribble", "action_release_dribble"]

	env = football_env.create_environment(
	env_name="11_vs_11_stochastic", representation="extracted", stacked=True,
	logdir='/content/logs', write_full_episode_dumps=True, write_video=True)

	cur_agent = Agent()
	memory_buffer = ReplayBuffer()
	old_agent = cur_agent
	env.reset()
	obs = env.getobs()
	policy_prob, values, rewards, states, actions, advantage = [], [], [], [], [], []
	for itr in range(1, ITERS):
		game_on = True
		for _ in range(1, TIMESTEPS):
			if game_on:
				policy, V_value = old_agent.predict(obs)
				action = tf.argmax(policy[0])
				next_obs, rew, done, info = env.step(action)
				if rew in [1, -1]:
					memory_buffer.add(obs, action, rew, V_value, max(policy[0]), True)
				else:
					memory_buffer.add(obs, action, rew, V_value, max(policy[0]), False)
		else:
			env.reset()
			_, value = old_agent.predict(obs)
			values.append(value)
			game_on = False
			# compute advantages
			advantages = get_advantages(rewards, values)

		# Early stopping check

if __name__ == "__main__":
	train()