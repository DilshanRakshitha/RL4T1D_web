import scipy.signal
import numpy as np
import pandas as pd
import logging
import gym
from gym.envs.registration import register
import warnings
import math
import torch
import pkg_resources


CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


def get_env(args, worker_id=None, env_type=None):

    patients, env_ids = get_patient_env()

    patient_name = patients[args.patient_id]
    # Ensure env_id is unique if multiple workers/sims run concurrently,
    # especially if gym doesn't unregister environments promptly.
    # Using a highly unique worker_id passed from app.py is good.
    # If worker_id is None (e.g. some local test script), generate one.
    unique_worker_id_str = str(worker_id if worker_id is not None else np.random.randint(10000, 20000))
    env_id = unique_worker_id_str + '_' + env_ids[args.patient_id]
    
    # Use args.seed which now carries the scenario_seed_override or a random one from get_simulation_args
    seed_to_use_for_env = args.seed 
    
    # Check if already registered, if so, skip. This can happen with fast restarts or overlapping calls.
    # A more robust solution might involve unregistering, but that can also be tricky.
    # For now, let's assume unique env_id per call due to api_worker_id in app.py is sufficient.
    # if env_id not in gym.envs.registry.env_specs: # This check might not be thread-safe or always reliable
    try:
        register(
            id=env_id,
            entry_point='environment.extended_T1DSimEnv:T1DSimEnv',
            kwargs={'patient_name': patient_name,
                    'reward_fun': custom_reward,
                    'seed': seed_to_use_for_env, # IMPORTANT: Use the seed from args
                    'args': args,
                    'env_type': env_type}
        )
    except gym.error.Error as e:
        if "Cannot re-register id" in str(e):
            flask_logger = logging.getLogger(__name__) # Get flask logger if available
            flask_logger.warning(f"Environment ID {env_id} already registered. Attempting to use existing.")
        else:
            raise e # Re-raise other gym errors

    env = gym.make(env_id)
    env_conditions = {'insulin_min': env.action_space.low, 'insulin_max': env.action_space.high,
                      'cgm_low': env.observation_space.low, 'cgm_high': env.observation_space.high}
    logging.info(f"Env {env_id} created/retrieved with patient {patient_name}, seed {seed_to_use_for_env}. Conditions: {env_conditions}")
    return env


def get_patient_env():
    patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
    env_ids = (['simglucose-adolescent{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-child{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-adult{}-v0'.format(str(i)) for i in range(1, 11)])
    return patients, env_ids


def get_patient_index(patient_type=None):
    low_index, high_index = -1, -1
    if patient_type == 'adult':
        low_index, high_index = 20, 29
    elif patient_type == 'child':
        low_index, high_index = 10, 19
    elif patient_type == 'adolescent':
        low_index, high_index = 0, 9
    else:
        # Consider logging an error or raising a ValueError if patient_type is unexpected
        logging.error(f'Unknown patient_type in get_patient_index: {patient_type}')
    return low_index, high_index


def risk_index(BG, horizon):
    # BG is in mg/dL, horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = np.array(BG[-horizon:])
        BG_to_compute[BG_to_compute < 1] = 1 # Avoid log(0)
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return LBGI, HBGI, RI


def custom_reward(bg_hist, **kwargs):
    # bg_hist is expected to be a list or array of BG values
    if not bg_hist: # Handle empty bg_hist case
        return 0 # Or some other default/error value
    return -risk_index([bg_hist[-1]], 1)[-1] # Use only the most recent BG value


def get_basal(patient_name='none'):
    if patient_name == 'none':
        logging.error('Patient name not provided to get_basal')
        return 0.0 # Default or raise error
    try:
        quest = pd.read_csv(CONTROL_QUEST)
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        q_row = quest[quest.Name.str.fullmatch(patient_name)] # Use fullmatch for exact name
        params_row = patient_params[patient_params.Name.str.fullmatch(patient_name)]

        if q_row.empty or params_row.empty:
            logging.error(f"Patient data not found for '{patient_name}' in Quest or Params files.")
            # Fallback for patient names like 'patient0', 'patient20' if direct match fails
            # This part is heuristic and might need adjustment based on actual naming in CSVs
            numeric_id_match = patient_name.replace("patient", "adolescent#0").replace("child#0", "child#0").replace("adult#0", "adult#0") # Basic attempt
            if "#0" not in numeric_id_match and patient_name.isdigit(): # e.g. "0" -> "adolescent#001"
                 num_id = int(patient_name)
                 if 0 <= num_id <= 9: numeric_id_match = f"adolescent#0{num_id+1:02d}"
                 elif 10 <= num_id <= 19: numeric_id_match = f"child#0{num_id-10+1:02d}"
                 elif 20 <= num_id <= 29: numeric_id_match = f"adult#0{num_id-20+1:02d}"

            q_row = quest[quest.Name.str.fullmatch(numeric_id_match)]
            params_row = patient_params[patient_params.Name.str.fullmatch(numeric_id_match)]
            if q_row.empty or params_row.empty:
                 logging.error(f"Fallback patient data not found for '{numeric_id_match}' either.")
                 return 0.0


        u2ss = params_row.u2ss.values.item()
        BW = params_row.BW.values.item()
        basal = u2ss * BW / 6000
    except Exception as e:
        logging.error(f"Error calculating basal for {patient_name}: {e}")
        return 0.0 # Default or raise error
    return basal