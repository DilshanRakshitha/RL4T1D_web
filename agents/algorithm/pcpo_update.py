import torch
import torch.nn as nn
from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic
from utils.onpolicy_buffers import RolloutBuffer
from utils.logger import LogExperiment
from utils.core import get_flat_params_from, set_flat_params_to, compute_flat_grad
import numpy as np
import scipy.optimize as opt


class PCPO(Agent):
    def __init__(self, args, env_args, load_model, actor_path, critic_path):
        super(PCPO, self).__init__(args, env_args=env_args)
        self.args = args
        self.env_args = env_args
        self.device = args.device
        self.completed_interactions = 0

        
        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        
        self.policy = ActorCritic(args, load_model, actor_path, critic_path).to(self.device)
        if args.verbose:
            print('PolicyNet Params: {}'.format(sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('ValueNet Params: {}'.format(sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=args.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=args.vf_lr)
        self.value_criterion = nn.MSELoss()

        self.RolloutBuffer = RolloutBuffer(args)

        # Safety constraints and parameters
        self.max_constraint_violation = args.max_constraint_violation
        self.target_kl = args.target_kl

        # Logging
        self.model_logs = torch.zeros(7, device=self.device)
        self.LogExperiment = LogExperiment(args)

    def train_pi(self):
        print('Running PCPO Policy Update...')

        # Policy optimization with constraint satisfaction
        for i in range(self.args.n_pi_epochs):
            states_batch = self.rollout_buffer['states']
            actions_batch = self.rollout_buffer['action']
            logprobs_batch = self.rollout_buffer['log_prob_action']
            advantages_batch = self.rollout_buffer['advantage']
            cost_adv_batch = self.rollout_buffer['cost_advantage']

            # Normalize the advantages and cost advantages
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
            cost_adv_batch = (cost_adv_batch - cost_adv_batch.mean()) / (cost_adv_batch.std() + 1e-5)

            logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
            ratios = torch.exp(logprobs_prediction - logprobs_batch).squeeze()

            # Policy loss and constraint satisfaction
            r_theta = ratios * advantages_batch
            policy_loss = -r_theta.mean() - self.args.entropy_coef * dist_entropy.mean()

            # Project the policy update to satisfy the constraint
            cost_theta = ratios * cost_adv_batch
            cost_loss = -cost_theta.mean()
            print("cost loss: ", cost_loss, " max constraint violation: ",self.max_constraint_violation)
            # If cost constraint is violated, perform projection
            if cost_loss > self.max_constraint_violation:
                print("cost loss is higher")
                # Projection mechanism to maintain feasible updates
                projected_step = self.project_policy_update(policy_loss, cost_loss)
                set_flat_params_to(self.policy.Actor, projected_step)

            self.optimizer_Actor.zero_grad()
            policy_loss.backward()
            self.optimizer_Actor.step()
        print("policy_loss: ",policy_loss)
        return policy_loss

    def project_policy_update(self, policy_loss, cost_loss):
        """
        Implement projection of policy updates to ensure constraint satisfaction.
        Typically, this involves solving a quadratic programming problem where
        the update direction is projected into the feasible region.
        """
        # Get current policy parameters
        prev_params = get_flat_params_from(self.policy.Actor)
        
        # Compute the gradients for policy loss
        grads = torch.autograd.grad(policy_loss, self.policy.Actor.parameters(), retain_graph=True)
        flat_grad = torch.cat([grad.view(-1) for grad in grads]).detach().cpu().numpy()

        # Define the objective function for quadratic programming
        def objective_fn(step_direction):
            return 0.5 * np.dot(step_direction, step_direction)  # Minimize step direction norm (Euclidean distance)
        
        # Define constraint: cost_loss <= max_constraint_violation
        def constraint_fn(step_direction):
            return cost_loss.item() + np.dot(flat_grad, step_direction) - self.max_constraint_violation

        # Set up bounds for the step direction (if necessary)
        bounds = [(None, None) for _ in flat_grad]

        # Solve the quadratic programming problem to project the update
        result = opt.minimize(
            objective_fn,
            np.zeros_like(flat_grad),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_fn}
        )

        # Extract the optimized step direction and apply it to the current parameters
        step_direction = result.x
        projected_step = prev_params + torch.tensor(step_direction).to(self.device)

        return projected_step

    def train_vf(self):
        print('Running Value Function Update...')

        val_loss_log, value_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)

        for i in range(self.args.n_vf_epochs):
            start_idx = 0
            while start_idx < self.rollout_buffer['len']:
                end_idx = min(start_idx + self.args.batch_size, self.rollout_buffer['len'])

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                value_target = self.rollout_buffer['value_target'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                value_prediction = self.policy.evaluate_critic(states_batch)
                value_loss = self.value_criterion(value_prediction, value_target)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.args.grad_clip)
                self.optimizer_Critic.step()

                val_loss_log += value_loss.detach()
                start_idx += self.args.batch_size

        return value_grad / val_count, val_loss_log

    def update(self):
        self.rollout_buffer = self.RolloutBuffer.prepare_rollout_buffer()
        print("rollout buffer done")
        self.model_logs[0] = self.train_pi()
        print("pi done")
        self.model_logs[1], self.model_logs[2] = self.train_vf()
        self.LogExperiment.save(log_name='/model_log', data=[self.model_logs.detach().cpu().flatten().numpy()])

# In the main function or RL loop:
# if __name__ == '__main__':
#     args = Args()  # Define or load your args object
#     env_args = EnvArgs()  # Define or load environment args
#     agent = PCPO(args, env_args, load_model=False, actor_path=None, critic_path=None)
#     agent.update()
