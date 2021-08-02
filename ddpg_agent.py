import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer


class Agent():
    def __init__(self,
                 alpha,
                 beta,
                 tau,
                 gamma,
                 input_dims,
                 n_actions,
                 max_size,
                 batch_size,
                 fc1_dims,
                 fc2_dims,
                 action_limit,):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.action_limit = action_limit

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  action_limit, n_actions=n_actions,
                                  name='actor')

        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')

        """ Initialize target networks """

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         action_limit, n_actions=n_actions,
                                         name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims,
                                           fc2_dims, n_actions=n_actions,
                                           name='target_critic')

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.update_network_parameters(tau=1)

    """ Select action a_t u(s_t|theta-u) + N_t according to current policy
        and exploration noise"""

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state).to(self.actor.device)
        action_noise = action + T.tensor(self.noise(),
                                         dtype=T.float).to(self.actor.device)
        return action_noise.cpu().detach().numpy()[0]  # !!!!!!! Action may need to be clip to a min, max value

    """ Store transition (s_t,a_t,r_t,s_t+1) into R"""

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    """Update policy and value parameters using given batch of experience tuples.

    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    where:
        actor_target(state) -> action
        critic_target(state, action) -> Q-value

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        """ Sample reandom minibatch of N transitions (s_t,a_t,r_t,s_t+1) from R"""

        states, actions, rewards, next_states, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(
            next_states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        """ Compute the target (y_i = r_i + gama*critic_val_next) """

        target_actions = self.target_actor.forward(next_states)  # action next
        critic_val_next = self.target_critic.forward(
            next_states, target_actions)  # Q_target_next
        critic_val = self.critic.forward(states, actions)  # Q_expected

        critic_val_next[done] = 0.0
        critic_val_next = critic_val_next.view(-1)

        target = rewards + self.gamma*critic_val_next  # Q_target
        target = target.view(self.batch_size, 1)

        """ Update the critic by minimizing the loss """

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_val)  # compute critic loss
        critic_loss.backward()  # backpropagate
        self.critic.optimizer.step()

        """ Update the actor by policy using the sample gradient """

        self.actor.optimizer.zero_grad()
        action_pred = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, action_pred)  # compute the actor loss
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()  # backpropagate
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        """ Update target networks parameters
        
        Soft update model parameters:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() +\
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        # self.target_critic.load_state_dict(critic_state_dict, strict = False)
        # self.target_actor.load_state_dict(actor_state_dict, stric=False)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
