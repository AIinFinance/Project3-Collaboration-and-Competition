# Learning Algorithm
As the Unity Reacher environment of the second project, the Unity Tennis environment is a multi-agent problem. I will use DDPG agent in this project as in the second project. However, there is a subtle difference in the two projects. While agents in the second project are independent and identical, agents in this project interact with each other. Thus, unlike in the second project, agents in this project cannot be reduced to a single agent problem. Thus, I made another file named `multi_agents.py’ in addition to ‘ddpg_agent.py’ in order to deal with multiple agents. The key part of `multi_agents.py’ is 
<p align="center">
[Agent(state_size, action_size, random_seed) for i in range(num_agents)]
</p>  

which is a list of individual ddpg_agents. Moreover, since all the agents share their experience, I moved `ReplayBuffer` which was originally in ‘ddpg_agent.py’ to `multi_agents.py’

# Plot of Rewards

<p align="center">
<img width="50%" src="https://user-images.githubusercontent.com/95396618/144891000-3ac0b681-c0ea-4d00-8d9d-8ccd72e23d3d.PNG"/>  
</p>  


<p align="center">
<img width="60%" src="https://user-images.githubusercontent.com/95396618/144890997-91ce3c03-a63b-4214-a323-3296f67a312f.PNG"/>  
</p>  

 


# Ideas for Future Work
As in the second project, I want to try soft actor-critic method for future work. Unlike DDPG, which use deterministic policies, soft actor-critic uses a stochastic policy. Moreover, soft actor-critic model works in a similar manner to temporal difference approach. The critic in soft actor-critic model uses both the Q function and the value function. Thus, it might provide better performance compared to DDPG.
