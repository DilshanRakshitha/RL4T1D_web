import torch
import torch.nn as nn
from agents.algorithm.agent import Agent
from agents.models.actor_critic import ActorCritic
from utils.onpolicy_buffers import RolloutBuffer
from utils.logger import LogExperiment
from utils.core import get_flat_params_from, set_flat_params_to, compute_flat_grad


class CPO(Agent):
    def __init__(self, args, env_args, load_model, actor_path, critic_path):
        super(CPO, self).__init__(args, env_args=env_args)
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
        print('Running Policy Update...')

        # conjugate gradient decent
        def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
            x = torch.zeros(b.size(), device=b.device)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            for i in range(nsteps):
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
        for i in range(self.train_pi_iters):
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

                # print(i)
                logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
                ratios = torch.exp(logprobs_prediction - logprobs_batch)
                ratios = ratios.squeeze()
                # r_theta = ratios * advantages_batch
                # + self.policy.Actor.PolicyModule.penalty * 0.00001
                # policy_loss = -r_theta.mean() - self.entropy_coef * dist_entropy.mean() 
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()

                # early stop: approx kl calculation
                log_ratio = logprobs_prediction - logprobs_batch
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
                if approx_kl > 1.5 * self.target_kl:
                    if self.args.verbose:
                        print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, n_batch, approx_kl))
                    continue_pi_training = False
                    break

                if torch.isnan(policy_loss):  # for debugging only!
                    print('policy loss: {}'.format(policy_loss))
                    exit()

                temp_loss_log += policy_loss.detach()
                policy_grad = torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                # policy_grad = torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), max_norm=1.0)
                policy_grad_ += policy_grad # used to returing mean_pi_gradient at the end
                # grads = torch.autograd.grad(policy_loss, self.policy.Actor.parameters(), retain_graph=True)
                # loss_grad = torch.cat([grad.view(-1) for grad in grads])
                # implement gradient normalizing if want here

                # Zero gradients, perform a backward pass, and compute the gradients
                self.optimizer_Actor.zero_grad()
                policy_loss.backward(retain_graph=True)

                # Extract the gradients computed by Adam
                grads = []
                for param in self.policy.Actor.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                loss_grad = torch.cat(grads)
                # finding the step direction / add direct hessian finding function here later. get the parameter from args
                Fvp = Fvp_fim
                stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
                # if gradient normalizing, normalize the step dir here

                # findign cost loss
                # c_theta = ratios * cost_advantages_batch
                # + self.policy.Actor.PolicyModule.penalty * 0.00001
                # cost_loss = -c_theta.mean() - self.entropy_coef * dist_entropy.mean()
                surr1_cost = ratios * cost_advantages_batch
                surr2_cost = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * cost_advantages_batch
                cost_loss = -torch.min(surr1_cost, surr2_cost).mean() - self.entropy_coef * dist_entropy.mean()



                #finding the cost step direction
                cost_grads = torch.autograd.grad(cost_loss, self.policy.Actor.parameters())
                cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]) 
                # Zero gradients, perform a backward pass, and compute the gradients for cost loss
                # self.optimizer_Actor.zero_grad()
                # cost_loss.backward()

                # Extract the gradients computed by Adam for cost loss
                # cost_grads = []
                # for param in self.policy.Actor.parameters():
                #     if param.grad is not None:
                #         cost_grads.append(param.grad.view(-1))
                # cost_loss_grad = torch.cat(cost_grads)
                cost_loss_grad = cost_loss_grad / torch.norm(cost_loss_grad)
                cost_stepdir = conjugate_gradients(Fvp, -cost_loss_grad, 10)

                # Define q, r, s
                p = -cost_loss_grad.dot(stepdir) #a^T.H^-1.g
                q = -loss_grad.dot(stepdir) #g^T.H^-1.g
                r = loss_grad.dot(cost_stepdir) #g^T.H^-1.a
                s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 

            
                epsilon = 1e-6
                s = s + epsilon
                self.d_k = torch.tensor(self.d_k).to(constraint.dtype).to(constraint.device)
                cc =  constraint - self.d_k
                lamda = 2*self.max_kl

                #find optimal lambda_a and  lambda_b
                A = torch.sqrt((q - (r**2)/s)/(self.max_kl - (cc**2)/s))
                B = torch.sqrt(q/self.max_kl)
        
                
                cc += epsilon
                if cc>0:
                    opt_lam_a = torch.max(r/cc,A)
                    opt_lam_b = torch.max(torch.zeros_like(A),torch.min(B,r/cc))
                else: 
                    
                    opt_lam_b = torch.max(r/cc,B)
                    
                    opt_lam_a = torch.max(torch.zeros_like(A),torch.min(A,r/cc))
                    
                
                #define f_a(\lambda) and f_b(\lambda)
                def f_a_lambda(lamda):
                    lamda = lamda + epsilon
                    a = ((r**2)/s - q)/(2*lamda)
                    b = lamda*((cc**2)/s - self.max_kl)/2
                    c = - (r*cc)/s
                    return a+b+c
                
                def f_b_lambda(lamda):
                    lamda = lamda + epsilon
                    a = -(q/lamda + lamda*self.max_kl)/2
                    return a   
                
                #find values of optimal lambdas 
                opt_f_a = f_a_lambda(opt_lam_a)
                opt_f_b = f_b_lambda(opt_lam_b)

                if opt_f_a > opt_f_b:
                    opt_lambda = opt_lam_a
                else:
                    opt_lambda = opt_lam_b
                        
                #find optimal nu
                nu = (opt_lambda*cc - r)/s 
                if nu>0:
                    opt_nu = nu 
                else:
                    opt_nu = 0

                # finding optimal step direction
                if ((cc**2)/s - self.max_kl) > 0 and cc>0:
                    print('cost exeeded')
                    opt_stepdir = torch.sqrt(2*self.max_kl/s)*Fvp(cost_stepdir)
                    prev_params = get_flat_params_from(self.policy.Actor)
                    new_params = prev_params + opt_stepdir
                    set_flat_params_to(self.policy.Actor, new_params)
                else:
                    # opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda
                    print('cost not exceeded')
                    self.optimizer_Actor.step()
                
                # trying without line search
                # prev_params = get_flat_params_from(self.policy.Actor)
                # new_params = prev_params + opt_stepdir
                # set_flat_params_to(self.policy.Actor, new_params)

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

    def update(self):
        self.rollout_buffer = self.RolloutBuffer.prepare_rollout_buffer()
        self.model_logs[0], self.model_logs[5] = self.train_pi()
        self.model_logs[1], self.model_logs[2], self.model_logs[3], self.model_logs[4] = self.train_vf()
        self.LogExperiment.save(log_name='/model_log', data=[self.model_logs.detach().cpu().flatten().numpy()])

def objective(trial):
    # Suggest hyperparameters using Optuna for d_k, max_kl, and damping
    d_k = trial.suggest_int('d_k', 10, 60)
    max_kl = trial.suggest_uniform('max_kl', 0.001, 0.05)
    damping = trial.suggest_uniform('damping', 0.01, 0.2)

    # Update the args object with the suggested hyperparameters
    args.d_k = d_k
    args.max_kl = max_kl
    args.damping = damping
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ensure the device is set

    # Instantiate the agent with updated hyperparameters
    agent = CPO(args, env_args, load_model=False, actor_path=None, critic_path=None)
    
    # Run the agent and evaluate its performance (e.g., average reward)
    reward = agent.run()  # Modify this to return the evaluation reward
    
    return reward  # Objective is to maximize reward

# Optuna hyperparameter tuning
if __name__ == '__main__':
    # Initialize your arguments and environment arguments
    args = Args()  # Replace with however you initialize your args object
    env_args = EnvArgs()  # Initialize env_args similarly

    # Set default values for args and env_args
    args.pi_lr = 1e-4  # Default values
    args.vf_lr = 1e-4
    args.entropy_coef = 0.01
    args.grad_clip = 0.5
    args.eps_clip = 0.2
    args.target_kl = 0.05
    args.n_vf_epochs = 5
    args.n_pi_epochs = 5
    args.batch_size = 64
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Save and print the best hyperparameters
    print(study.best_params)

    


