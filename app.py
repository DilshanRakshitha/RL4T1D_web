import os
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from argparse import Namespace
from datetime import datetime, timedelta

# --- RL Classes ---
from agents.models.actor_critic import ActorCritic # Policy network structure

# --- Environment Classes & Utils ---
import gym
from gym.envs.registration import register
# The modified T1DSimEnv that accepts custom scenarios
from environment.extended_T1DSimEnv import T1DSimEnv as ExtendedT1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient # Base patient class
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.env import T1DSimEnv as SimGlucoseBaseEnv # Simglucose's core env

from environment.reward_func import composite_reward
from environment.state_space import StateSpace
from utils.control_space import ControlSpace # Action conversion
from environment.utils import get_basal # To get std_basal for patient

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# Adjust these paths and parameters as needed
MODEL_DIR = "./models/"  # CREATE THIS DIRECTORY and place your models here
# Example: PPO_MODEL_FILENAME = "ppo_actor_critic_adolescent#001" (without .pth)
# The code will append _Actor.pth and _Critic.pth
PPO_MODEL_FILENAME = "ppo_model_for_patient_adolescent001" # Replace X with patient id like adolescent#001
PCPO_MODEL_FILENAME = "pcpo_model_for_patient_adolescent001" # Replace X (SRPO-like)

SIM_PATIENT_NAME = 'adolescent001' # Patient for simulation
SIM_DURATION_MINUTES = 24 * 60     # Duration of simulation (e.g., 1 day)
SIM_SAMPLING_RATE = 5              # minutes, typically 5 for simglucose

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created model directory: {MODEL_DIR}. Please place your trained models here.")
    print(f"Example PPO Actor: {MODEL_DIR}{PPO_MODEL_FILENAME}_Actor.pth")
    print(f"Example PCPO Actor: {MODEL_DIR}{PCPO_MODEL_FILENAME}_Actor.pth")

# --- Helper: Simplified Args Namespace ---
def get_simulation_args(agent_name_for_paths="sim_run"):
    args = Namespace()

    # Experiment settings
    args.experiment_folder = f"temp_web_sim_output/{agent_name_for_paths}"
    args.experiment_dir = args.experiment_folder
    os.makedirs(args.experiment_dir, exist_ok=True, mode=0o755) # Ensure path exists
    # For worker logs if any part tries to write:
    os.makedirs(os.path.join(args.experiment_dir, "testing", "data"), exist_ok=True, mode=0o755)


    args.device = "cpu"
    args.verbose = False # Can be True for debugging
    args.seed = random.randint(0, 10000)

    # Environment settings (from configs/env/env.yaml & configs/config.yaml)
    args.patient_name = SIM_PATIENT_NAME
    args.n_action = 1
    args.n_features = 2
    args.feature_history = 12
    args.calibration = 12
    args.control_space_type = "exponential"
    args.insulin_min = 0.0
    args.insulin_max = 5.0
    args.glucose_max = 600.0
    args.glucose_min = 39.0
    args.sensor = "GuardianRT"
    args.pump = "Insulet"
    args.t_meal = 20 # Minutes, for potential meal announcements if features were used
    args.sampling_rate = SIM_SAMPLING_RATE # Important for state updates and meal timings

    # Minimal features for inference (matching training)
    args.use_meal_announcement = False
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    
    # RL Agent Model Structure (from agent YAMLs, e.g. ppo.yaml)
    args.n_rnn_hidden = 16
    args.n_rnn_layers = 1
    args.rnn_directions = 1
    args.bidirectional = False

    # Params for reward and GAE (not strictly needed for inference if policy is fixed)
    args.return_type = 'average'
    args.gamma = 0.99
    args.lambda_ = 0.95
    args.normalize_reward = True

    # Simulation loop settings
    args.n_step = SIM_DURATION_MINUTES // SIM_SAMPLING_RATE
    args.max_epi_length = args.n_step
    args.max_test_epi_len = args.n_step # Used by worker/simulation loop

    # Dummy/default values for other params an Agent might expect
    args.debug = False
    args.pi_lr = 3e-4
    args.vf_lr = 3e-4
    # ... add other common params from your agent configs if they are accessed in ActorCritic or FeatureExtractor
    
    return args

# --- Helper: Environment Creation ---
def create_simulation_env(args, scenario_meal_data, worker_id_suffix="sim"):
    """
    Creates and registers a T1DSimEnv with a custom meal scenario.
    scenario_meal_data: list of (time_offset_minutes, carb_grams)
    """
    env_id = f'simglucose-{args.patient_name.replace("#", "")}-{worker_id_suffix}-v0'
    
    # CustomScenario expects meal times in hours from start_time
    start_time = datetime(2023, 1, 1, 0, 0, 0) # Arbitrary fixed start
    
    custom_scenario_struct = []
    if scenario_meal_data:
        for meal_time_min, meal_carbs in scenario_meal_data:
            custom_scenario_struct.append((meal_time_min / 60.0, meal_carbs))

    # Ensure environment is registered only once or handle re-registration
    if env_id not in gym.envs.registry.env_specs:
         register(
            id=env_id,
            entry_point='environment.extended_T1DSimEnv:T1DSimEnv', # Your modified Env
            kwargs={
                'patient_name': args.patient_name,
                'reward_fun': composite_reward,
                'seed': args.seed + hash(worker_id_suffix), # Unique seed
                'args': args,
                'env_type': 'testing', # Always testing mode for inference
                'custom_scenario_params': {
                    'start_time': start_time,
                    'scenario_list': custom_scenario_struct
                }
            }
        )
    env = gym.make(env_id)
    return env

# --- Helper: Simulation Execution ---
def execute_episode(policy_network, env, sim_args):
    cgm_trace, insulin_trace, meal_trace, reward_trace = [], [], [], []
    
    # The environment's reset() in ExtendedT1DSimEnv handles calibration
    # and returns the initial processed state for the policy.
    current_processed_state = env.reset() 
    state_space_handler = StateSpace(sim_args) # Used if we need to re-process, but env should provide it

    total_reward = 0.0
    # Standard basal for the patient, used by get_pump_action if control_space needs it
    std_basal = get_basal(patient_name=sim_args.patient_name)


    for t_step in range(sim_args.max_test_epi_len):
        with torch.no_grad():
            # Policy expects state processed by StateSpace (which env.reset() and env.step() should provide)
            policy_output = policy_network.get_action(current_processed_state)
        
        rl_action_normalized = policy_output['action']

        pump_action_value = controlspace.map(agent_action=rl_action_normalized['action'][0])

        pump_action_value = get_pump_action(
            agent_action=rl_action_normalized,
            control_space_type=sim_args.control_space_type,
            std_basal=std_basal,
            iob_scalar=1.0 # Default, typically not used by 'exponential'
        )
        
        # Environment step returns next_processed_state, reward, done, info
        next_processed_state, reward, done, info = env.step(pump_action_value)
        
        # Log data
        raw_cgm = info.get('cgm').CGM if info.get('cgm') else sim_args.glucose_max # Simglucose Observation object
        current_meal_input = info.get('meal', 0.0) * sim_args.sampling_rate # CHO g this step
        
        cgm_trace.append(float(raw_cgm))
        insulin_trace.append(float(pump_action_value))
        meal_trace.append(float(current_meal_input)) # This will be non-zero only at meal steps
        reward_trace.append(float(reward))
        
        current_processed_state = next_processed_state
        total_reward += reward

        if done:
            if sim_args.verbose: print(f"Simulation ended early at step {t_step} due to 'done' flag.")
            break
            
    return {
        "cgm": cgm_trace,
        "insulin": insulin_trace,
        "meals_input_per_step": meal_trace, # Frontend might need to aggregate this
        "rewards": reward_trace,
        "total_reward": total_reward,
        "patient_name": sim_args.patient_name,
        "duration_steps": len(cgm_trace)
    }

# --- Main Simulation Runner ---
def run_simulation_for_agent(model_filename_base, meal_data_for_sim):
    """
    Loads a policy model and runs a simulation with custom meal data.
    model_filename_base: e.g., "ppo_model_for_patient_X"
    meal_data_for_sim: list of (time_in_minutes, carb_grams)
    """
    sim_args = get_simulation_args(agent_name_for_paths=model_filename_base)

    actor_path = os.path.join(MODEL_DIR, f"{model_filename_base}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{model_filename_base}_Critic.pth")

    if not os.path.exists(actor_path):
        print(f"Warning: Actor model not found at {actor_path}")
        return {"error": f"Actor model not found: {actor_path}"}
    if not os.path.exists(critic_path): # Critic needed for ActorCritic structure
        print(f"Warning: Critic model not found at {critic_path}")
        return {"error": f"Critic model not found: {critic_path}"}

    # Load the ActorCritic policy network
    policy_net = ActorCritic(args=sim_args, load=True, actor_path=actor_path, critic_path=critic_path)
    policy_net.to(sim_args.device)
    policy_net.eval() # Set to evaluation mode

    # Create environment
    env = create_simulation_env(sim_args, meal_data_for_sim, worker_id_suffix=model_filename_base)
    
    # Execute simulation
    simulation_results = execute_episode(policy_net, env, sim_args)
    
    env.close() # Clean up gym environment
    return simulation_results

# --- Flask API Endpoint ---
@app.route('/simulate', methods=['POST'])
def simulate_glucose_endpoint():
    try:
        data = request.json
        if not data or 'meals' not in data:
            return jsonify({"error": "Missing 'meals' data in request."}), 400

        # meals: list of {"time_minutes": int, "carbs": float}
        meal_details_input = data.get('meals', []) 
        
        # Convert to format expected by simulation: [(time_offset_minutes, carb_grams), ...]
        parsed_meal_data = []
        for meal_item in meal_details_input:
            if "time_minutes" in meal_item and "carbs" in meal_item:
                parsed_meal_data.append((int(meal_item["time_minutes"]), float(meal_item["carbs"])))
            else:
                return jsonify({"error": "Each meal item must have 'time_minutes' and 'carbs'."}), 400
        
        # Sort meals by time just in case
        parsed_meal_data.sort(key=lambda x: x[0])

        if app.config.get('VERBOSE_DEBUG', False): # For debugging
            print(f"Received meal data: {parsed_meal_data}")

        ppo_results = run_simulation_for_agent(PPO_MODEL_FILENAME, parsed_meal_data)
        pcpo_results = run_simulation_for_agent(PCPO_MODEL_FILENAME, parsed_meal_data) # PCPO for "SRPO-like"

        return jsonify({
            "ppo_simulation": ppo_results,
            "pcpo_simulation": pcpo_results # "SRPO-like" results
        })

    except FileNotFoundError as e:
        app.logger.error(f"Model file not found: {e}")
        return jsonify({"error": f"A required model file was not found: {e}. Please check server configuration and model paths."}), 500
    except Exception as e:
        app.logger.error(f"Error during simulation: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.config['VERBOSE_DEBUG'] = True # Enable print statements for meal data
    app.run(debug=True, host='0.0.0.0', port=5000)