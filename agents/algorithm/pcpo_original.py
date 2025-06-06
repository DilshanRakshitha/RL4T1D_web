import torch
import torch.nn as nn
from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic
from utils.onpolicy_buffers import RolloutBuffer
from utils.logger import LogExperiment
from utils.core import get_flat_params_from, set_flat_params_to, compute_flat_grad


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

        # load models and setup optimiser.
        self.policy = ActorCritic(args, load_model, actor_path, critic_path).to(self.device)
        if args.verbose:
            print('PolicyNet Params: {}'.format(sum(p.numel() for p in self.policy.Actor.parameters() if p.requires_grad)))
            print('ValueNet Params: {}'.format(sum(p.numel() for p in self.policy.Critic.parameters() if p.requires_grad)))
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()

        self.RolloutBuffer = RolloutBuffer(args)
        self.rollout_buffer = {}

        # ppo params
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.target_kl = args.target_kl
        self.d_k = args.d_k
        self.max_kl = args.max_kl
        self.damping = args.damping
        # self.constraint = self.rollout_buffer['constraint']

        # logging
        self.model_logs = torch.zeros(7, device=self.args.device)
        self.LogExperiment = LogExperiment(args)
    

    def train_pi(self):
        print('Running CPO Policy Update...')

        # conjugate gradient decent
        def conjugate_gradients(Avp_f, b, nsteps=20, rdotr_tol=1e-10):
            x = torch.zeros(b.size(), device=b.device)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            # for i in range(nsteps):
            while rdotr >= rdotr_tol:
                Avp = Avp_f(p)
                alpha = rdotr / torch.dot(p, Avp)
                x += alpha * p
                r -= alpha * Avp
                new_rdotr = torch.dot(r, r)
                betta = new_rdotr / rdotr
                p = r + betta * p
                rdotr = new_rdotr
                if rdotr < rdotr_tol:
                    break
            return x
    
        # implementing fisher information matrix
        def Fvp_direct(v):
            kl = self.policy.Actor.get_kl(states_batch)
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.policy.Actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.policy.Actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

            return flat_grad_grad_kl + v * self.damping
    
        def Fvp_fim(v):
            with torch.backends.cudnn.flags(enabled=False):
                M, mu, info = self.policy.Actor.get_fim(states_batch)
                mu = mu.view(-1)
                filter_input_ids = set([info['std_id']])

                t = torch.ones(mu.size(), requires_grad=True, device=mu.device)
                mu_t = (mu * t).sum()
                Jt = compute_flat_grad(mu_t, self.policy.Actor.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
                Jtv = (Jt * v).sum()
                Jv = torch.autograd.grad(Jtv, t)[0]
                MJv = M * Jv.detach()
                mu_MJv = (MJv * mu).sum()
                JTMJv = compute_flat_grad(mu_MJv, self.policy.Actor.parameters(), filter_input_ids=filter_input_ids, create_graph=True).detach()
                JTMJv /= states_batch.shape[0]
                std_index = info['std_index']
                JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
                return JTMJv + v * self.damping

        temp_loss_log = torch.zeros(1, device=self.device)
        policy_grad, pol_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_pi_training, buffer_len = True, self.rollout_buffer['len']
        constraint = self.rollout_buffer['constraint']
        policy_grad_ = 0
        start_idx, n_batch = 0, 0
        while start_idx < buffer_len:
            n_batch += 1
            end_idx = min(start_idx + self.batch_size, buffer_len)

            states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
            actions_batch = self.rollout_buffer['action'][start_idx:end_idx, :]
            logprobs_batch = self.rollout_buffer['log_prob_action'][start_idx:end_idx, :]
            advantages_batch = self.rollout_buffer['advantage'][start_idx:end_idx]
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
            cost_advantages_batch = self.rollout_buffer['cost_advantage'][start_idx:end_idx]
            cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (cost_advantages_batch.std() + 1e-5)

            logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
            ratios = torch.exp(logprobs_prediction - logprobs_batch)
            ratios = ratios.squeeze()
            r_theta = ratios * advantages_batch
            # + self.policy.Actor.PolicyModule.penalty * 0.00001
            # policy_loss = -r_theta.mean() - self.entropy_coef * dist_entropy.mean() 
            policy_loss = -r_theta.mean()

            # early stop: approx kl calculation
            log_ratio = logprobs_prediction - logprobs_batch
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
            if approx_kl > 1.5 * self.target_kl:
                if self.args.verbose:
                    print('Early stop => Batch {}, Approximate KL: {}.'.format(n_batch, approx_kl))
                continue_pi_training = False
                break

            if torch.isnan(policy_loss):  # for debugging only!
                print('policy loss: {}'.format(policy_loss))
                exit()

            temp_loss_log += policy_loss.detach()
            policy_grad = torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
            # policy_grad = torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), max_norm=1.0)
            policy_grad_ += policy_grad # used to returing mean_pi_gradient at the end
            grads = torch.autograd.grad(policy_loss, self.policy.Actor.parameters(), retain_graph=True)
            loss_grad = torch.cat([grad.view(-1) for grad in grads])
            # implement gradient normalizing if want here

            # finding the step direction / add direct hessian finding function here later. get the parameter from args
            Fvp = Fvp_fim
            stepdir = conjugate_gradients(Fvp, -loss_grad, 10) #point which minimizes the loss
            # if gradient normalizing, normalize the step dir here

            # findign cost loss
            c_theta = ratios * cost_advantages_batch
            # + self.policy.Actor.PolicyModule.penalty * 0.00001
            """cost_loss = -c_theta.mean() - self.entropy_coef * dist_entropy.mean()
            initially had this, no need for the- sign in cost loss, as we are minimizing the cost function"""
            cost_loss = c_theta.mean()

            #finding the cost step direction
            cost_grads = torch.autograd.grad(cost_loss, self.policy.Actor.parameters())
            cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]) 
            # cost_loss_grad = cost_loss_grad / torch.norm(cost_loss_grad)
            cost_stepdir = conjugate_gradients(Fvp, -cost_loss_grad, 10) #point which minimizes the cost loss

            # Define q, r, s
            """p is feels to be worng -> p = cost_loss_grad.dot(stepdir)
                but since p is not being used, won't change it. """
            p = cost_loss_grad.dot(stepdir) #a^T.H^-1.g
            q = -loss_grad.dot(stepdir) #g^T.H^-1.g
            r = loss_grad.dot(cost_stepdir) #g^T.H^-1.a
            s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 

            print('s =', s, 'r =', r, 'q =', q, 'p =', p)
            epsilon = 1e-8
            policy_update = torch.sqrt(2*self.max_kl/(q + epsilon))*stepdir
            project_val = ((torch.sqrt(2*self.max_kl/(q + epsilon)) * p) + (constraint - self.d_k)) / (s + epsilon)
            print('(torch.sqrt(2*self.max_kl/(q + epsilon)) * p) = ',(torch.sqrt(2*self.max_kl/(q + epsilon)) * p))
            print('(constraint - self.d_k)) / (s + epsilon) = ', ((constraint - self.d_k)) / (s + epsilon))
            print('self.d_k = ', self.d_k)
            print('constraint - self.d_k = ', constraint - self.d_k)
            print("project_val = ", project_val.item())
            projection = (max(0.0, project_val.item())) * cost_stepdir
            opt_step = policy_update + projection
            
            # trying without line search
            print("constraint =", constraint)
            prev_params = get_flat_params_from(self.policy.Actor)
            new_params = prev_params + opt_step
            set_flat_params_to(self.policy.Actor, new_params)

            #######
            pol_count += 1
            start_idx += self.batch_size

            if not continue_pi_training:
                break
        mean_pi_grad = policy_grad_ / pol_count if pol_count != 0 else 0
        print('The policy loss is: {}'.format(temp_loss_log))
        return mean_pi_grad, temp_loss_log

    def train_vf(self):
        print('Running Value Function Update...')

        # variables to be logged for debugging purposes.
        val_loss_log, value_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        true_var, explained_var = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)

        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < self.rollout_buffer['len']:
                end_idx = min(start_idx + self.batch_size, self.rollout_buffer['len'])

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                value_target = self.rollout_buffer['value_target'][start_idx:end_idx]

                self.optimizer_Critic.zero_grad()
                value_prediction = self.policy.evaluate_critic(states_batch)
                value_loss = self.value_criterion(value_prediction, value_target)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)  # clip gradients before optimising
                self.optimizer_Critic.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = value_prediction.detach().flatten()
                y_true = value_target.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)

        return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count
    
    def train_vf_cost(self):
        print('Running Cost Value Function Update...')

        # variables to be logged for debugging purposes.
        val_loss_log, value_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        true_var, explained_var = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        val_count = torch.zeros(1, device=self.device)

        for i in range(self.train_v_iters):
            start_idx = 0
            while start_idx < self.rollout_buffer['len']:
                end_idx = min(start_idx + self.batch_size, self.rollout_buffer['len'])

                states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                value_target = self.rollout_buffer['cost_val_target'][start_idx:end_idx]

                self.optimizer_Critic_cost.zero_grad()
                value_prediction = self.policy.evaluate_costcritic(states_batch)
                value_loss = self.value_criterion(value_prediction, value_target)
                value_loss.backward()
                value_grad += torch.nn.utils.clip_grad_norm_(self.policy.CriticCost.parameters(), self.grad_clip)  # clip gradients before optimising
                self.optimizer_Critic_cost.step()
                val_count += 1
                start_idx += self.batch_size

                # logging.
                val_loss_log += value_loss.detach()
                y_pred = value_prediction.detach().flatten()
                y_true = value_target.flatten()
                var_y = torch.var(y_true)
                true_var += var_y
                explained_var += 1 - torch.var(y_true - y_pred) / (var_y + 1e-5)

        # return value_grad / val_count, val_loss_log, explained_var / val_count, true_var / val_count
        return

    def update(self):
        self.rollout_buffer = self.RolloutBuffer.prepare_rollout_buffer()
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4] = self.train_vf()
        self.train_vf_cost()
        self.LogExperiment.save(log_name='/model_log', data=[self.model_logs.detach().cpu().flatten().numpy()])


