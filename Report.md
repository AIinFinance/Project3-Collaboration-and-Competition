# Learning Algorithm
As the Unity Reacher environment in the project 2, the Unity Tennis environment is a multi-agent problem. I will use DDPG agent in this project as in the project 2. However, there is a subtle difference in the two projects. While agents in the project 2 are independent and identical, agents in this project interact with each other. Thus, unlike in the project 2, agents in this project cannot be reduced to a single agent problem. Thus, I made another file named `multi_agents.py’ in addition to ‘ddpg_agent.py’. Key part of `multi_agents.py’ is 
[Agent(state_size, action_size, random_seed) for i in range(num_agents)]
which is a list of individual ddpg_agent. Moreover, since all the agents share their experience, I moved `ReplayBuffer` which was originally in ‘ddpg_agent.py’ to `multi_agents.py’

# Plot of Rewards

# Ideas for Future Work
