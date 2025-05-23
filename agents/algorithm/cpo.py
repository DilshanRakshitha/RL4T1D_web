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
        print('Running CPO Policy Update...')

        # success, new_params = line_search(self.policy.Actor, policy_loss, prev_params, fullstep, expected_improve)
        def line_search(states_batch, actions_batch, logprobs_batch, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
            
            def get_loss():
                with torch.set_grad_enabled(False):
                    logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
                    ratios = torch.exp(logprobs_prediction - logprobs_batch)
                    ratios = ratios.squeeze()
                    r_theta = ratios * advantages_batch
                    # + self.policy.Actor.PolicyModule.penalty * 0.00001
                    policy_loss = -r_theta.mean() - self.entropy_coef * dist_entropy.mean() 
                    return policy_loss

            fval = get_loss()
            for stepfrac in [.5**x for x in range(max_backtracks)]:
                x_new = x + stepfrac * fullstep
                set_flat_params_to(self.policy.Actor, x_new)
                fval_new = get_loss()
                actual_improve = fval - fval_new
                expected_improve = expected_improve_full * stepfrac
                ratio = actual_improve / expected_improve

                if ratio > accept_ratio:
                    return True, x_new
            return False, x

        # conjugate gradient decent
        def conjugate_gradients(Avp_f, b, nsteps=20, rdotr_tol=1e-10):
            x = torch.zeros(b.size(), device=b.device)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            for i in range(nsteps):
            # while rdotr>=rdotr_tol:
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
            # start_idx, n_batch = 0, 0
            # while start_idx < buffer_len:
            # n_batch += 1
            # end_idx = min(start_idx + self.batch_size, buffer_len)

            states_batch = self.rollout_buffer['states'][:, :, :]
            actions_batch = self.rollout_buffer['action'][:, :]
            logprobs_batch = self.rollout_buffer['log_prob_action'][:, :]
            advantages_batch = self.rollout_buffer['advantage'][:]
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-5)
            cost_advantages_batch = self.rollout_buffer['cost_advantage'][:]
            cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (cost_advantages_batch.std() + 1e-5)

            logprobs_prediction, dist_entropy = self.policy.evaluate_actor(states_batch, actions_batch)
            ratios = torch.exp(logprobs_prediction - logprobs_batch)
            ratios = ratios.squeeze()
            r_theta = ratios * advantages_batch
            # + self.policy.Actor.PolicyModule.penalty * 0.00001
            policy_loss = -r_theta.mean() - self.entropy_coef * dist_entropy.mean() 

            # early stop: approx kl calculation
            log_ratio = logprobs_prediction - logprobs_batch
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).detach().cpu().numpy()
            if approx_kl > 1.5 * self.target_kl:
                if self.args.verbose:
                    print('Early stop => Epoch {}, Batch {}, Approximate KL: {}.'.format(i, 1, approx_kl))
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
            p = -cost_loss_grad.dot(stepdir) #a^T.H^-1.g
            q = -loss_grad.dot(stepdir) #g^T.H^-1.g
            r = loss_grad.dot(cost_stepdir) #g^T.H^-1.a
            s = -cost_loss_grad.dot(cost_stepdir) #a^T.H^-1.a 
        
            epsilon = 1e-8
            self.d_k = torch.tensor(self.d_k).to(constraint.dtype).to(constraint.device)
            cc =  constraint - self.d_k

            """"""
            
            A = q - r**2 / s
            B = self.max_kl - cc**2 / s
            optim_case = -1
            if(cc < 0 and B <0):
                optim_case = 3
            elif (cc < 0 and B > 0):
                optim_case = 2
            elif (cc > 0 and B > 0):
                optim_case = 1
            else:
                optim_case = 0

            lam = torch.sqrt(q / self.max_kl)
            nu = 0

            if (optim_case == 2 or optim_case == 1):
                lam_mid = r / cc
                L_mid = -0.5 * (q / lam_mid + lam_mid * self.max_kl)

                lam_a = torch.sqrt(A / (B + epsilon))
                L_a = -torch.sqrt(A * B) - r * cc / (s + epsilon)
                print('A = ', A, 'B = ', B)
                lam_b = torch.sqrt(q / self.max_kl)
                L_b = - torch.sqrt( q * self.max_kl)
                # print('lam_a = ', lam_a, 'lam_b = ', lam_b, 'lam_mid = ', lam_mid)
                # print('L_a = ', L_a, 'L_b = ', L_b, 'L_mid = ', L_mid)
                if lam_mid > 0:
                    if cc < 0:
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid
                    else:
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid
                    
                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b
                else:
                    if lam_a > 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * cc - r) / (s + epsilon)
            # print('lam = ', lam, 'nu = ', nu)
            if (optim_case > 0):
                opt_stepdir = -(1. / (lam + epsilon) ) * ( stepdir - nu * cost_stepdir )
                # opt_stepdir = (stepdir + opt_nu*cost_stepdir)/opt_lambda
            else:
                opt_stepdir = torch.sqrt(self.max_kl / (s + epsilon)) * cost_stepdir
                # opt_stepdir = torch.sqrt(2*self.max_kl/s)*cost_stepdir
            # -------
            # print('opt_stepdir = ', opt_stepdir)
            print('constraint = ', constraint)

            """"""
            # find optimal lambda_a and  lambda_b
            # print('q = ', q, 'r = ', r, 's = ', s, 'cc = ', cc)
            # print('(q - (r**2)/s) = ', (q - (r**2)/s), '(self.max_kl - (cc**2)/s) = ', (self.max_kl - (cc**2)/s))
            # A = torch.sqrt((q - (r**2)/s)/(self.max_kl - (cc**2)/s))
            # B = torch.sqrt(q/self.max_kl)
            
            # cc += epsilon
            # s = s + epsilon
            # lamda = 2*self.max_kl
            # if cc>0:
            #     opt_lam_a = torch.max(r/cc,A)
            #     opt_lam_b = torch.max(torch.zeros_like(A),torch.min(B,r/cc))
            #     print('opt_lam_a = ', opt_lam_a, 'opt_lam_b = ', opt_lam_b)
            # else: 
            #     opt_lam_b = torch.max(r/cc,B)
            #     print('torch.zeros_like(A) = ', torch.zeros_like(A), 'torch.min(A,r/cc) = ', torch.min(A,r/cc))
            #     print('A = ', A, 'r/cc = ', r/cc)
            #     opt_lam_a = torch.max(torch.zeros_like(A),torch.min(A,r/cc))
            #     print('opt_lam_a = ', opt_lam_a, 'opt_lam_b = ', opt_lam_b)
            
            # #define f_a(\lambda) and f_b(\lambda)
            # def f_a_lambda(lamda):
            #     lamda = lamda + epsilon
            #     a = ((r**2)/s - q)/(2*lamda)
            #     b = lamda*((cc**2)/s - self.max_kl)/2
            #     c = - (r*cc)/s
            #     print('lambda = ', lamda , 's = ' , s ,'a = ', a, 'b = ', b, 'c = ',c )
            #     return a+b+c
            
            # def f_b_lambda(lamda):
            #     lamda = lamda + epsilon
            #     a = -(q/lamda + lamda*self.max_kl)/2
            #     print('a = ', a)
            #     return a   
            
            # #find values of optimal lambdas 
            # opt_f_a = f_a_lambda(opt_lam_a)
            # opt_f_b = f_b_lambda(opt_lam_b)

            # if opt_f_a > opt_f_b:
            #     opt_lambda = opt_lam_a
            # else:
            #     opt_lambda = opt_lam_b
                    
            # #find optimal nu
            # nu = (opt_lambda*cc - r)/s 
            # if nu>0:
            #     opt_nu = nu 
            # else:
            #     opt_nu = 0

            # # finding optimal step direction
            # if ((cc**2)/s - self.max_kl) > 0 and cc>0:
            #     print('updated cost exeeded')
            #     print('constraint = ', constraint)
            #     """opt_stepdir = torch.sqrt(2*self.max_kl/s)*Fvp(cost_stepdir)
            #         previously had the above, but check the paper, correct implementation is this. ig"""
            #     opt_stepdir = torch.sqrt(self.max_kl/s)*cost_stepdir
            # else:
            #     print('updated cost not exceeded')
            #     print('constraint = ', constraint)
            #     """opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda
            #     this is th earlier version but since, cost_step_dir = - H^-1.b, we should have the below"""
            #     print('step_dir = ', stepdir, "opt_nu = ", opt_nu, "cost_stepdir = ", cost_stepdir, "opt_lambda = ", opt_lambda)
            #     opt_stepdir = (stepdir + opt_nu*cost_stepdir)/(opt_lambda + epsilon)
            
            # trying without line search
            prev_params = get_flat_params_from(self.policy.Actor)
            new_params = prev_params + opt_stepdir
            set_flat_params_to(self.policy.Actor, new_params)

            # trying with Line search
            #find the maximum step length
            # xhx = opt_stepdir.dot(Fvp(opt_stepdir))
            # beta_1 = -cc/(cost_loss_grad.dot(opt_stepdir))
            # beta_2 = torch.sqrt(self.max_kl / xhx)
            
            # if beta_1 < beta_2:
            #     beta_star = beta_1
            # else: 
            #     beta_star = beta_2
            
            # # perform line search
            # #fullstep = beta_star*opt_stepdir
            # prev_params = get_flat_params_from(self.policy.Actor)
            # fullstep = opt_stepdir
            # expected_improve = -loss_grad.dot(fullstep)
            # success, new_params = line_search(states_batch, actions_batch, logprobs_batch, prev_params, fullstep, expected_improve)
            # set_flat_params_to(self.policy.Actor, new_params)

            # def line_search(states_batch, actions_batch, logprobs_batch, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    
            
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

        if(self.completed_interactions % 440960 == 0):
            self.d_k -= 5


