import gc
import abc
import time
import torch
import random

from metrics.metrics import time_in_range
from metrics.statistics import calc_stats
from utils.worker import OnPolicyWorker as Worker
from utils.core import get_flat_params_from, set_flat_params_to, compute_flat_grad

from decouple import config
MAIN_PATH = config('MAIN_PATH')

import pandas as pd
# import wandb


class Agent:
    def __init__(self, args, env_args):
        self.args = None
        self.env_args = None
        self.policy = None
        self.RolloutBuffer = None

        # workers run the simulations. For each worker an env is created, and the worker ID should be unique.
        self.n_training_workers = args.n_training_workers
        self.n_testing_workers = args.n_testing_workers
        self.total_interactions = args.total_interactions
        self.n_interactions_lr_decay = args.n_interactions_lr_decay
        self.n_val_trials = args.n_val_trials

        # The offset params above are for visual convenience of raw logs when going through worker logs which are saved as:
        # e.g., worker_10.csv, worker_5000.csv, workers with 5000+ are testing; workers with 6000+ are validation
        self.training_agent_id_offset = 5  # 5, 6, 7, ... (5+n_training_workers)
        self.testing_agent_id_offset = 5000  # 5000, 5001, 5002, ... (5000+n_testing_workers)
        self.validation_agent_id_offset = 6000  # 6000, 6001, 6002, ... (6000+n_val_trials)
        self.completed_interactions = 0
        self.best = 0
        self.current = 0
        self.best_normo = 0
        self.current_normo = 0
        self.best_params = None
        self.final_best_params = None
        self.temperature = 1

        if args.debug:
            self.n_testing_workers = 2
            self.n_training_workers = 2
            self.total_interactions = 4000
            self.n_interactions_lr_decay = 2000
            self.n_val_trials = 3

    @abc.abstractmethod
    def update(self):
        """
        Implement the update rule.
        """

    def run(self):
        # initialise workers for training
        training_agents = [Worker(args=self.args, env_args=self.env_args, mode='training', worker_id=i+self.training_agent_id_offset)
                           for i in range(self.n_training_workers)]

        # initialise workers for testing after each update step
        testing_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i+self.testing_agent_id_offset)
                          for i in range(self.n_testing_workers)]

        # start ppo learning
        rollout, self.completed_interactions = 0, 0
        while self.completed_interactions < self.total_interactions:  # steps * n_workers * epochs. 3000 is just a large number
            tstart = time.perf_counter()
            # run training workers to collect data
            for i in range(self.n_training_workers):
                training_agents[i].rollout(policy=self.policy, buffer=self.RolloutBuffer.RolloutWorker)
                self.RolloutBuffer.save_rollout(training_agent_index=i)
            self.update()  # update the models
            self.policy.save(rollout)  # save model weights as checkpoints

            # testing: run testing workers on the validation scenario
            with torch.no_grad():
                counter_list = []
                normo_list = []
                for i in range(self.n_testing_workers):
                    counter, normo = testing_agents[i].rollout(policy=self.policy, buffer=None)  # these logs will be saved by the worker.
                    counter_list.append(counter)
                    normo_list.append(normo)
                
            counter_mean = sum(counter_list) / len(counter_list)
            normo_mean = sum(normo_list)/ len(normo_list)
            self.current = counter_mean
            self.current_normo =  normo_mean
            if(counter_mean >= self.best):
                # if(counter_mean > self.best):
                #     self.best_normo = normo_mean
                # elif(counter_mean == self.best and normo_mean >= self.best_normo):
                #     self.best_normo = normo_mean
                #     self.final_best_params = get_flat_params_from(self.policy.Actor)
                self.best = counter_mean
                self.best_params = get_flat_params_from(self.policy.Actor)
                

            randnum = random.random()
            current_params = get_flat_params_from(self.policy.Actor)
            print('randnum: {}, temperature: {}, avg_t: {}, best_avg_t: {}, avg_normo: {}, best_avg_normo: {}.'.format(randnum, self.temperature, self.current, self.best, self.current_normo, self.best_normo))
            if(self.completed_interactions > 400000  and self.current <= self.best and self.best_params != None and not torch.equal(self.best_params, current_params)):
                self.temperature *= 0.99
                if(randnum > self.temperature):
                    print('Early stop => randnum: {}, temperature: {}, avg_t: {}, best_avg_t: {}, avg_normo: {}, best_avg_normo: {}.'.format(randnum, self.temperature, self.current, self.best, self.current_normo, self.best_normo))
                    set_flat_params_to(self.policy.Actor, self.best_params)

            # update the total number of completed interactions.
            self.completed_interactions += (self.args.n_step * self.n_training_workers)
            rollout += 1
            # print('completed interactions', self.completed_interactions)
            gc.collect()  # garbage collector to clean unused objects.

            # decay lr and set entropy coeff to zero to stabilise the policy towards the end.
            if self.completed_interactions == self.n_interactions_lr_decay:
                self.decay_lr()

            experiment_done = True if self.completed_interactions > self.total_interactions else False

            # logging
            #wandb.log({"Training Progress": (completed_interactions/self.total_interactions)*100})
            print('\n---------------------------------------------------------')
            print('Training Progress: {:.2f}%, Elapsed time: {:.4f} minutes.'.format(min(100.00, (self.completed_interactions/self.total_interactions)*100),
                                                                                     (time.perf_counter() - tstart)/60))
            print('---------------------------------------------------------')

            # print('Rollout Time (seconds): {}, update: {}, testing: {}'.format((t2 - t1), (t4 - t2), (t6 - t4))) if self.args.verbose else None
            # self.LogExperiment.save(log_name='/experiment_summary', data=[[experiment_done, rollout, (t2 - t1), (t4 - t2), (t6 - t4)]])

            # when training complete conduct final validation: typically n=500.
            if experiment_done:
                set_flat_params_to(self.policy.Actor, self.best_params)
                self.evaluate()

    def evaluate(self):
        print('\n---------------------------------------------------------')
        print('===> Starting Validation Trials ....')
        validation_agents = [Worker(args=self.args, env_args=self.env_args, mode='testing', worker_id=i + self.validation_agent_id_offset)
                             for i in range(self.n_val_trials)]
        with torch.no_grad():
            for i in range(self.n_val_trials):
                validation_agents[i].rollout(policy=self.policy, buffer=None)

            # calculate the final metrics.
            cohort_res, summary_stats = [], []
            secondary_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
            data = []
            FOLDER_PATH = '/results/'+self.args.experiment_folder+'/testing/data'
            for i in range(0, self.n_val_trials):
                test_i = 'logs_worker_'+str(self.validation_agent_id_offset+i)+'.csv'
                df = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i)
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'])
                reward_val = df['rew'].sum()*(100/288)
                e = [[i, df.shape[0], reward_val, normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper, 0, 0]]
                dataframe=pd.DataFrame(e, columns=secondary_columns)
                data.append(dataframe)
            res = pd.concat(data)
            res['PatientID'] = self.args.patient_id
            res.rename(columns={'sev_hypo':'S_hypo', 'sev_hyper':'S_hyper'}, inplace=True)
            summary_stats.append(res)
            metric=['mean', 'std', 'min', 'max']
            print(calc_stats(res, metric=metric, sim_len=288))

            print('\nAlgorithm Training/Validation Completed Successfully.')
            print('---------------------------------------------------------')
            exit()

    def decay_lr(self):
        self.entropy_coef = 0  # self.entropy_coef / 100
        self.pi_lr = self.pi_lr / 10
        self.vf_lr = self.vf_lr / 10
        for param_group in self.optimizer_Actor.param_groups:
            param_group['lr'] = self.pi_lr
        for param_group in self.optimizer_Critic.param_groups:
            param_group['lr'] = self.vf_lr