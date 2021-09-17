import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")


LEARING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EVERY = 100

epslion = 0.5
START_EPSILON = 1
END_EPSILON = EPISODES // 2 
epslion_deacy_vaule = epslion / (END_EPSILON - START_EPSILON)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_reward = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discete(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for e in range(EPISODES):
	episodes_reward = 0
	if e % SHOW_EVERY == 0:
		render = True
	else:
		render = False
	discrete_state = get_discete(env.reset())
	done = False
	while not done:
		if np.random.random()  > epslion:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, _ = env.step(action)
		episodes_reward += reward
		new_discrete_state = get_discete(new_state)
		if render:
			env.render()
		if not done:
			max_futer_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - LEARING_RATE) * current_q + LEARING_RATE * (reward + DISCOUNT * max_futer_q) 
			q_table[discrete_state+(action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print(f"reached!!!!!!!!!!!!!!!!{e}")
			number_of_succ += 1
			q_table[discrete_state + (action, )] = 0
		discrete_state = new_discrete_state
	if END_EPSILON >= e >= START_EPSILON:
		epslion -= epslion_deacy_vaule
	ep_reward.append(episodes_reward)

	if not e % SHOW_EVERY:
		average_reward = sum(ep_reward[- SHOW_EVERY:]) / len(ep_reward[- SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(e)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_reward[- SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_reward[- SHOW_EVERY:]))

env.close()




plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')

plt.legend(loc = 4)
plt.show()
