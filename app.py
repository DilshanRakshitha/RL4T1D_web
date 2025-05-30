import os
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from argparse import Namespace
from datetime import datetime
import logging
import shutil
import time
import threading

# --- Add warning suppression ---
import warnings
from pkg_resources import PkgResourcesDeprecationWarning
warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning, module="gym.envs.registration")
# --- End warning suppression ---

from agents.models.actor_critic import ActorCritic # Assuming this is your model class
from utils.worker import OnPolicyWorker # Assuming this is your worker class

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'a_very_long_random_and_unique_secret_key_for_this_app_v8_extreme_resilience_daily' # Updated
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=50)


flask_logger = app.logger
if not flask_logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')
    flask_logger = logging.getLogger(__name__)


MODEL_DIR = "./models/"
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}"
SRPO_MODEL_FILENAME_BASE = "srpo_model_for_patient_{}"
SWITCHING_MODEL_FILENAME_BASE = "switching_model_for_patient_{}"

MODEL_FILENAME_BASES = {
    "ppo": PPO_MODEL_FILENAME_BASE,
    "srpo": SRPO_MODEL_FILENAME_BASE,
    "switching": SWITCHING_MODEL_FILENAME_BASE
}

SIM_DURATION_MINUTES = 24 * 60
SIM_SAMPLING_RATE = 5
# Min steps for a day to be considered "successful" for scenario seed continuity
MIN_STEPS_FOR_SUCCESSFUL_DAY = (SIM_DURATION_MINUTES // SIM_SAMPLING_RATE) - 10 

VALID_PATIENT_IDS = [str(i) for i in range(9)] + [str(i) for i in range(20, 30)]
VALID_COMPARISON_MODES = ["ppo_vs_srpo", "ppo_vs_switching", "srpo_vs_switching"]
ALGORITHMS = ["ppo", "srpo", "switching"] 

running_simulations_stop_events = {}
client_meal_schedules = {}


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Modified to accept scenario_seed_override
def get_simulation_args(run_identifier_prefix="sim_run", patient_id_str="0", scenario_seed_override=None):
    args = Namespace()
    unique_log_id = f"{run_identifier_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id)
    args.experiment_dir = args.experiment_folder
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)
    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', False)
    
    # Use override if provided, else generate a new random seed for the scenario/environment
    args.seed = scenario_seed_override if scenario_seed_override is not None else random.randint(0, 1000000)
    
    project_root_for_env_util = os.path.dirname(os.path.abspath(__file__))
    main_path_val_util = os.environ.get('MAIN_PATH', project_root_for_env_util)
    original_main_path = os.environ.get('MAIN_PATH')
    if not original_main_path:
        os.environ['MAIN_PATH'] = main_path_val_util
    from environment.utils import get_patient_env 
    patients_from_util_with_hash, _ = get_patient_env()
    if not original_main_path:
        del os.environ['MAIN_PATH']
    elif original_main_path != os.environ.get('MAIN_PATH'):
        os.environ['MAIN_PATH'] = original_main_path
    
    try:
        args.patient_id = int(patient_id_str)
        if not (0 <= args.patient_id < len(patients_from_util_with_hash)):
            flask_logger.warning(f"Patient ID {args.patient_id} (from '{patient_id_str}') out of bounds. Defaulting to 0.")
            args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[args.patient_id]
        args.patient_name = patient_id_str
    except Exception as e: 
        flask_logger.error(f"Error resolving patient_id_str '{patient_id_str}': {e}. Defaulting to patient_id 0.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "patient#000"
        args.patient_name = "0"

    args.n_action = 1; args.n_features = 2; args.feature_history = 12; args.calibration = 12
    args.control_space_type = "exponential"; args.insulin_min = 0.0; args.insulin_max = 5.0
    args.glucose_max = 600.0; args.glucose_min = 39.0; args.sensor = "GuardianRT"; args.pump = "Insulet"
    args.t_meal = 20; args.sampling_rate = SIM_SAMPLING_RATE
    args.use_meal_announcement = False; args.use_carb_announcement = False; args.use_tod_announcement = False
    args.n_rnn_hidden = 16; args.n_rnn_layers = 1; args.rnn_directions = 1; args.bidirectional = False
    args.return_type = 'average'; args.gamma = 0.99; args.lambda_ = 0.95; args.normalize_reward = True
    args.n_step = SIM_DURATION_MINUTES // SIM_SAMPLING_RATE 
    args.max_epi_length = args.n_step 
    args.max_test_epi_len = args.n_step 
    args.debug = False; args.pi_lr = 3e-4; args.vf_lr = 3e-4 
    
    num_meal_slots = 6 
    args.meal_times_mean = [0.0] * num_meal_slots
    args.time_variance = [1e-8] * num_meal_slots 
    args.time_lower_bound = [0.0] * num_meal_slots
    args.time_upper_bound = [0.0] * num_meal_slots
    args.meal_prob = [-1.0] * num_meal_slots 
    args.meal_amount = [0.0] * num_meal_slots
    args.meal_variance = [1e-8] * num_meal_slots
    args.val_meal_prob = [-1.0] * num_meal_slots
    args.val_meal_amount = [0.0] * num_meal_slots
    args.val_meal_variance = [1e-8] * num_meal_slots
    args.val_time_variance = [1e-8] * num_meal_slots
    return args

def configure_custom_scenario_args(base_args, scenario_meal_data_tuples):
    worker_args = Namespace(**vars(base_args)) 
    custom_meal_times_hours = []
    custom_meal_amounts_grams = []
    if scenario_meal_data_tuples:
        for meal_time_min, meal_carbs in scenario_meal_data_tuples:
            custom_meal_times_hours.append(float(meal_time_min) / 60.0) 
            custom_meal_amounts_grams.append(float(meal_carbs))
    num_meal_slots = len(worker_args.meal_times_mean) 
    cfg_meal_times_mean = [0.0] * num_meal_slots
    cfg_val_meal_amount = [0.0] * num_meal_slots
    cfg_val_meal_prob = [-1.0] * num_meal_slots 
    for i in range(len(custom_meal_times_hours)):
        if i < num_meal_slots:
            cfg_meal_times_mean[i] = custom_meal_times_hours[i]
            cfg_val_meal_amount[i] = custom_meal_amounts_grams[i]
            cfg_val_meal_prob[i] = 1.0 
        else:
            flask_logger.warning(f"Meal entry {i+1} (time: {custom_meal_times_hours[i]}h) exceeds {num_meal_slots} slots. Ignoring.")
    worker_args.meal_times_mean = cfg_meal_times_mean
    worker_args.val_meal_amount = cfg_val_meal_amount
    worker_args.val_meal_prob = cfg_val_meal_prob
    worker_args.val_time_variance = [1e-8] * num_meal_slots 
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(cfg_meal_times_mean) 
    worker_args.time_upper_bound = list(cfg_meal_times_mean)
    worker_args.env_type = 'testing' 
    return worker_args

# Modified to use scenario_seed_override and return scenario_seed_used
def run_simulation_segment(client_sid, model_filename_base, meal_data_tuples, agent_type_tag, sim_id_segment, stop_event, patient_name_str_override, scenario_seed_override=None):
    flask_logger.info(f"Enter run_simulation_segment for {agent_type_tag}, patient {patient_name_str_override}, segment ID: {sim_id_segment}, using scenario seed: {scenario_seed_override if scenario_seed_override is not None else 'NEW RANDOM'}")
    actual_model_filename_partial = model_filename_base.format(patient_name_str_override)
    base_args = get_simulation_args(
        run_identifier_prefix=sim_id_segment,
        patient_id_str=patient_name_str_override,
        scenario_seed_override=scenario_seed_override
    )
    worker_args = configure_custom_scenario_args(base_args, meal_data_tuples)
    scenario_seed_used_for_this_segment = worker_args.seed # This is the seed used for gym.make -> scenario

    original_main_path_temp_env = os.environ.get('MAIN_PATH')
    if not original_main_path_temp_env: os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
    from environment.utils import get_patient_env
    patients_list_full, _ = get_patient_env()
    if not original_main_path_temp_env: del os.environ['MAIN_PATH']
    elif original_main_path_temp_env != os.environ.get('MAIN_PATH'): os.environ['MAIN_PATH'] = original_main_path_temp_env

    if not (0 <= worker_args.patient_id < len(patients_list_full)):
        err_msg = f"Invalid patient_id {worker_args.patient_id} for {agent_type_tag}. Available: {len(patients_list_full)}."
        flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag, 'fatal':True}, room=client_sid); return {"error": err_msg, 'scenario_seed_used': scenario_seed_used_for_this_segment}
    worker_args.patient_name_for_env = patients_list_full[worker_args.patient_id]
    actor_path = os.path.join(MODEL_DIR, f"patient{patient_name_str_override}", f"{actual_model_filename_partial}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"patient{patient_name_str_override}", f"{actual_model_filename_partial}_Critic.pth")
    if not os.path.exists(actor_path):
        err_msg = f"Actor model not found: {actor_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag, 'fatal':True}, room=client_sid); return {"error": err_msg, 'scenario_seed_used': scenario_seed_used_for_this_segment}
    if not os.path.exists(critic_path):
        err_msg = f"Critic model not found: {critic_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag, 'fatal':True}, room=client_sid); return {"error": err_msg, 'scenario_seed_used': scenario_seed_used_for_this_segment}
    sim_results = {}; 
    original_main_path_worker_ctx = os.environ.get('MAIN_PATH') 
    if not original_main_path_worker_ctx: os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
    try:
        policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
        policy_net.to(worker_args.device); policy_net.eval()
        # Ensure worker_id is highly unique for each call to prevent gym registration clashes if not cleaned up fast enough
        api_worker_id = abs(hash(sim_id_segment + agent_type_tag + str(time.time()) + str(random.random()))) % 10000 + 7000 
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id) 
        worker.rollout(policy=policy_net, buffer=None) 
        if stop_event.is_set(): return None # Important: return None if stopped during rollout
        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        if not os.path.exists(log_file_path):
            err_msg = f"Simulation log file not found: {log_file_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag, 'fatal':True}, room=client_sid); return {"error": err_msg, 'scenario_seed_used': scenario_seed_used_for_this_segment}
        df = pd.read_csv(log_file_path)
        if df.empty:
            flask_logger.warning(f"Simulation log file is empty: {log_file_path} for {agent_type_tag}.")
            return {'cgm': [], 'insulin': [], 'meal': [], 'rewards': [], 'patient_name': worker_args.patient_name, 'duration_steps': 0, 'scenario_seed_used': scenario_seed_used_for_this_segment}
        required_cols = ['cgm', 'ins', 'rew', 'meals_input_per_step']
        for col in required_cols:
            if col not in df.columns:
                if col == 'meals_input_per_step': df[col] = 0.0 
                else: 
                    err_msg = f"Missing required column '{col}' in log file: {log_file_path}"
                    flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag, 'fatal':True}, room=client_sid); return {"error": err_msg, 'scenario_seed_used': scenario_seed_used_for_this_segment}
        if df[['cgm', 'ins', 'rew']].isnull().any().any(): flask_logger.warning(f"NaN values found in critical columns in {log_file_path}.")
        sim_results = {
            'cgm': df['cgm'].tolist(), 'insulin': df['ins'].tolist(), 'meal': df['meals_input_per_step'].tolist(), 
            'rewards': df['rew'].tolist(), 'patient_name': worker_args.patient_name, 
            'duration_steps': len(df), 'scenario_seed_used': scenario_seed_used_for_this_segment
        }
    except Exception as e:
        flask_logger.error(f"EXCEPTION during simulation segment for {agent_type_tag}: {e}", exc_info=True)
        return {"error": f"Exception occurred in {agent_type_tag} simulation segment: {str(e)}", 'scenario_seed_used': scenario_seed_used_for_this_segment}
    finally:
        if not original_main_path_worker_ctx and 'MAIN_PATH' in os.environ: del os.environ['MAIN_PATH']
        elif original_main_path_worker_ctx and os.environ.get('MAIN_PATH') != original_main_path_worker_ctx: os.environ['MAIN_PATH'] = original_main_path_worker_ctx
        folder_to_cleanup = getattr(base_args, 'experiment_folder', None)
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try: shutil.rmtree(folder_to_cleanup)
            except Exception as e_clean: flask_logger.error(f"Error cleaning temp dir {folder_to_cleanup}: {e_clean}")
    return sim_results

def emit_meal_markers_for_view(client_sid, current_global_start_step, steps_in_day_view):
    if client_sid not in client_meal_schedules:
        flask_logger.warning(f"Meal schedule not found for client {client_sid} when emitting markers.")
        return
    meal_schedule_for_client = client_meal_schedules[client_sid]
    markers = []
    
    # Determine the range of days to show markers for, relative to the current data's start
    current_day_idx_in_view = current_global_start_step // steps_in_day_view
    
    for day_offset in range(-1, 2): # Show markers for previous, current, and next visual day block
        target_day_idx = current_day_idx_in_view + day_offset
        if target_day_idx < 0: continue # Don't show markers for days before day 0

        day_base_step = target_day_idx * steps_in_day_view # Global step for the start of this target day
        
        for meal_info_dict in meal_schedule_for_client:
            meal_time_min_in_day = meal_info_dict['time_minutes'] # Time of meal within its 24h cycle
            meal_carbs = meal_info_dict['carbs']

            if not (0 <= meal_time_min_in_day < SIM_DURATION_MINUTES):
                flask_logger.warning(f"Meal time {meal_time_min_in_day} for client {client_sid} is out of bounds for a day. Skipping marker.")
                continue
            
            meal_step_within_day = meal_time_min_in_day // SIM_SAMPLING_RATE
            global_meal_step = day_base_step + meal_step_within_day
            
            markers.append({'step': global_meal_step, 'carbs': meal_carbs})
            
    socketio.emit('meal_markers', {'markers': markers}, room=client_sid)


# Major refactor for continuous daily simulation
def run_continuous_simulation_loop(client_sid, meal_data_list_of_dicts, stop_event, algorithms_to_compare_list, selected_patient_id_str):
    algo_names_str = " & ".join(algo.upper() for algo in algorithms_to_compare_list)
    flask_logger.info(f"BG Task for {client_sid}: Comparing {algo_names_str}, Patient: {selected_patient_id_str}.")
    
    day_number = 0 
    global_step_offset = 0 # Tracks the beginning of the current day in global steps
    
    if client_sid not in client_meal_schedules: # Should be set by handler, but good fallback
        client_meal_schedules[client_sid] = meal_data_list_of_dicts

    algo1_name = algorithms_to_compare_list[0]
    algo2_name = algorithms_to_compare_list[1]
    
    # Store the last seed that resulted in a successful full day simulation for scenario continuity
    algo1_last_successful_scenario_seed = None 
    algo2_last_successful_scenario_seed = None

    try:
        # Initial setup to get patient name, sampling rate, etc. Use a random seed here as it's pre-loop.
        temp_args_for_name = get_simulation_args(f"init_name_{client_sid}", patient_id_str=selected_patient_id_str, scenario_seed_override=random.randint(0,1000000))
        patient_name_for_sim_display = temp_args_for_name.patient_name
        initial_sampling_rate = temp_args_for_name.sampling_rate
        steps_per_day_segment = SIM_DURATION_MINUTES // initial_sampling_rate
        del temp_args_for_name
    except Exception as e: # Catch errors during this initial setup
        flask_logger.error(f"BG Task {client_sid}: Error determining patient name/rate: {e}", exc_info=True)
        socketio.emit('simulation_error', {'error': 'Initialization failed: Could not determine patient information.', 'fatal': True}, room=client_sid)
        # Ensure cleanup if this initial part fails
        if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules: del client_meal_schedules[client_sid]
        return

    socketio.emit('simulation_metadata', {
        'patient_name': patient_name_for_sim_display,
        'algorithms_compared': algorithms_to_compare_list,
        'sampling_rate': initial_sampling_rate
    }, room=client_sid)

    # Emit initial meal markers based on global_step_offset = 0
    emit_meal_markers_for_view(client_sid, global_step_offset, steps_per_day_segment)
    meal_data_tuples_for_worker = [(m['time_minutes'], m['carbs']) for m in meal_data_list_of_dicts]
    algo1_model_base = MODEL_FILENAME_BASES[algo1_name]
    algo2_model_base = MODEL_FILENAME_BASES[algo2_name]
    
    session_had_a_day_where_both_algos_failed = False

    try: # Wrap the main simulation loop
        while not stop_event.is_set():
            day_number += 1 
            flask_logger.info(f"BG Task {client_sid}: DAY {day_number} PROCESSING START. Patient: {selected_patient_id_str}. Global offset: {global_step_offset}")
            sim_id_segment_base = f"sid_{client_sid}_comp_{algo1_name}-{algo2_name}_p_{selected_patient_id_str}_day_{day_number}_glob_{global_step_offset}"

            # Initialize data for the current day
            algo1_data_this_day = {'duration_steps': 0, 'cgm': [], 'insulin': [], 'rewards': [], 'meal': [], 'scenario_seed_used': None, 'error_message': None}
            duration1_this_day = 0
            algo1_failed_this_day_flag = False 
            
            algo2_data_this_day = {'duration_steps': 0, 'cgm': [], 'insulin': [], 'rewards': [], 'meal': [], 'scenario_seed_used': None, 'error_message': None}
            duration2_this_day = 0
            algo2_failed_this_day_flag = False

            # --- Run Algo 1 for this day ---
            socketio.emit('simulation_progress', {'message': f'Computing Day {day_number} for {algo1_name.upper()}...'}, room=client_sid)
            temp_data1 = run_simulation_segment(client_sid, algo1_model_base, meal_data_tuples_for_worker, algo1_name, f"{sim_id_segment_base}_{algo1_name}", stop_event, selected_patient_id_str, algo1_last_successful_scenario_seed)
            
            if stop_event.is_set(): flask_logger.info(f"Stop event after trying to run {algo1_name} for Day {day_number}."); break
            
            if isinstance(temp_data1, dict) and "error" not in temp_data1:
                algo1_data_this_day = temp_data1
                duration1_this_day = algo1_data_this_day.get('duration_steps', 0)
                if duration1_this_day < MIN_STEPS_FOR_SUCCESSFUL_DAY: 
                    algo1_failed_this_day_flag = True
                    algo1_data_this_day['error_message'] = f"{algo1_name.upper()} did not complete Day {day_number} (ran {duration1_this_day}/{steps_per_day_segment} steps)."
                    flask_logger.warning(f"BG Task {client_sid}: {algo1_data_this_day['error_message']}")
                    socketio.emit('simulation_warning', {'warning': algo1_data_this_day['error_message'], 'agent_type_tag': algo1_name}, room=client_sid)
                    algo1_last_successful_scenario_seed = None # Failed, so reset seed for next attempt
                else:
                    algo1_last_successful_scenario_seed = algo1_data_this_day.get('scenario_seed_used', None) # Success, carry over seed
            elif isinstance(temp_data1, dict) and "error" in temp_data1: 
                algo1_failed_this_day_flag = True 
                algo1_data_this_day['error_message'] = temp_data1.get('error', f'Unknown error in {algo1_name} on Day {day_number}.')
                flask_logger.error(f"BG Task {client_sid}: Critical error for {algo1_name}: {algo1_data_this_day['error_message']}. This algo failed Day {day_number}.")
                socketio.emit('simulation_warning', {'warning': f"Critical error in {algo1_name.upper()}: {algo1_data_this_day['error_message']}. This algo failed Day {day_number}.", 'agent_type_tag': algo1_name}, room=client_sid)
                algo1_last_successful_scenario_seed = None
            # No `elif temp_data1 is None and stop_event.is_set()` here because we check stop_event right after the call
            else: # Should not happen if run_simulation_segment always returns dict or None on stop
                algo1_failed_this_day_flag = True
                algo1_data_this_day['error_message'] = f'Unexpected server result from {algo1_name} on Day {day_number}.'
                flask_logger.error(f"BG Task {client_sid}: {algo1_data_this_day['error_message']}")
                socketio.emit('simulation_warning', {'warning': algo1_data_this_day['error_message'], 'agent_type_tag': algo1_name}, room=client_sid)
                algo1_last_successful_scenario_seed = None
            flask_logger.info(f"BG Task {client_sid}: {algo1_name.upper()} Day {day_number} Duration: {duration1_this_day}. Failed this day: {algo1_failed_this_day_flag}. Next seed: {algo1_last_successful_scenario_seed}")
            
            if stop_event.is_set(): break # Check again before algo2

            # --- Run Algo 2 for this day ---
            socketio.emit('simulation_progress', {'message': f'Computing Day {day_number} for {algo2_name.upper()}...'}, room=client_sid)
            temp_data2 = run_simulation_segment(client_sid, algo2_model_base, meal_data_tuples_for_worker, algo2_name, f"{sim_id_segment_base}_{algo2_name}", stop_event, selected_patient_id_str, algo2_last_successful_scenario_seed)

            if stop_event.is_set(): flask_logger.info(f"Stop event after trying to run {algo2_name} for Day {day_number}."); break

            if isinstance(temp_data2, dict) and "error" not in temp_data2:
                algo2_data_this_day = temp_data2
                duration2_this_day = algo2_data_this_day.get('duration_steps', 0)
                if duration2_this_day < MIN_STEPS_FOR_SUCCESSFUL_DAY:
                    algo2_failed_this_day_flag = True
                    algo2_data_this_day['error_message'] = f"{algo2_name.upper()} did not complete Day {day_number} (ran {duration2_this_day}/{steps_per_day_segment} steps)."
                    flask_logger.warning(f"BG Task {client_sid}: {algo2_data_this_day['error_message']}")
                    socketio.emit('simulation_warning', {'warning': algo2_data_this_day['error_message'], 'agent_type_tag': algo2_name}, room=client_sid)
                    algo2_last_successful_scenario_seed = None
                else:
                    algo2_last_successful_scenario_seed = algo2_data_this_day.get('scenario_seed_used', None)
            elif isinstance(temp_data2, dict) and "error" in temp_data2:
                algo2_failed_this_day_flag = True
                algo2_data_this_day['error_message'] = temp_data2.get('error', f'Unknown error in {algo2_name} on Day {day_number}.')
                flask_logger.error(f"BG Task {client_sid}: Critical error for {algo2_name}: {algo2_data_this_day['error_message']}. This algo failed Day {day_number}.")
                socketio.emit('simulation_warning', {'warning': f"Critical error in {algo2_name.upper()}: {algo2_data_this_day['error_message']}. This algo failed Day {day_number}.", 'agent_type_tag': algo2_name}, room=client_sid)
                algo2_last_successful_scenario_seed = None
            else:
                algo2_failed_this_day_flag = True
                algo2_data_this_day['error_message'] = f'Unexpected server result from {algo2_name} on Day {day_number}.'
                flask_logger.error(f"BG Task {client_sid}: {algo2_data_this_day['error_message']}")
                socketio.emit('simulation_warning', {'warning': algo2_data_this_day['error_message'], 'agent_type_tag': algo2_name}, room=client_sid)
                algo2_last_successful_scenario_seed = None
            flask_logger.info(f"BG Task {client_sid}: {algo2_name.upper()} Day {day_number} Duration: {duration2_this_day}. Failed this day: {algo2_failed_this_day_flag}. Next seed: {algo2_last_successful_scenario_seed}")
            
            if stop_event.is_set(): break
            
            # --- Stream Data ---
            # Stream up to the max duration of the two, even if one or both failed short
            num_steps_to_stream_this_day = max(duration1_this_day, duration2_this_day)
            
            emit_meal_markers_for_view(client_sid, global_step_offset, steps_per_day_segment)
            if num_steps_to_stream_this_day > 0:
                socketio.emit('simulation_progress', {'message': f'Day {day_number} results. Streaming {num_steps_to_stream_this_day} steps...'}, room=client_sid)
            else: 
                socketio.emit('simulation_progress', {'message': f'Day {day_number}: No new data to stream from either algorithm.'}, room=client_sid)

            for i in range(num_steps_to_stream_this_day): # Stream only the actual data generated
                if stop_event.is_set(): flask_logger.info(f"Stop during streaming Day {day_number} step {i}."); break
                current_global_step = global_step_offset + i
                try:
                    data_point = {'step': current_global_step, 'algo1_tag': algo1_name, 'algo2_tag': algo2_name}
                    
                    # Algo1 data for this step i
                    if i < duration1_this_day: # Data exists from algo1 for this step
                        data_point.update({
                            'algo1_cgm': algo1_data_this_day['cgm'][i], 
                            'algo1_insulin': algo1_data_this_day['insulin'][i], 
                            'algo1_reward': algo1_data_this_day['rewards'][i]
                        })
                        if 'meal' not in data_point: # Prioritize meal from algo1 if it ran
                             data_point['meal'] = algo1_data_this_day['meal'][i]
                    else: # No data from algo1 for this step (it finished earlier or failed)
                        data_point.update({'algo1_cgm': None, 'algo1_insulin': None, 'algo1_reward': None}) 
                    
                    # Algo2 data for this step i
                    if i < duration2_this_day: # Data exists from algo2 for this step
                        data_point.update({
                            'algo2_cgm': algo2_data_this_day['cgm'][i], 
                            'algo2_insulin': algo2_data_this_day['insulin'][i], 
                            'algo2_reward': algo2_data_this_day['rewards'][i]
                        })
                        if 'meal' not in data_point: # Set meal from algo2 if not already set
                             data_point['meal'] = algo2_data_this_day['meal'][i]
                    else: # No data from algo2
                        data_point.update({'algo2_cgm': None, 'algo2_insulin': None, 'algo2_reward': None})
                    
                    if 'meal' not in data_point: data_point['meal'] = 0 # Ensure meal key is always present

                    socketio.emit('simulation_data_point', data_point, room=client_sid)
                    socketio.sleep(0.005) 
                except IndexError: 
                    flask_logger.error(f"IndexError streaming Day {day_number}. Global:{current_global_step}, Local:{i}. Dur1:{duration1_this_day}, Dur2:{duration2_this_day}", exc_info=True)
                    socketio.emit('simulation_error', {'error': 'Internal server error during data streaming. Session stopped.', 'fatal': True}, room=client_sid)
                    stop_event.set(); break 
                except Exception as e_stream:
                    flask_logger.error(f"Error emitting data Day {day_number}, Global:{current_global_step}: {e_stream}", exc_info=True)
                    socketio.emit('simulation_error', {'error': f'Server error during data streaming: {e_stream}. Session stopped.', 'fatal': True}, room=client_sid)
                    stop_event.set(); break
            if stop_event.is_set(): break # Exit while loop if stop event was set during streaming

            # --- Log if both algorithms failed this specific day (but do not terminate session based on this) ---
            if algo1_failed_this_day_flag and algo2_failed_this_day_flag:
                session_had_a_day_where_both_algos_failed = True 
                error_message_for_log = f"On Day {day_number}: {algo1_name.upper()} failed (Reason: {algo1_data_this_day.get('error_message', 'N/A')}) AND {algo2_name.upper()} failed (Reason: {algo2_data_this_day.get('error_message', 'N/A')}). Session will continue."
                flask_logger.warning(f"BG Task {client_sid}: {error_message_for_log}")
                socketio.emit('simulation_warning', {'warning': f"Both algorithms failed to complete Day {day_number}. The simulation will attempt to continue to Day {day_number + 1}.", 'agent_type_tag': 'session'}, room=client_sid)

            # --- Advance global step offset by a full day for visual separation ---
            global_step_offset += steps_per_day_segment
            
            if not stop_event.is_set():
                socketio.emit('simulation_progress', {'message': f"Day {day_number} processed. Preparing for Day {day_number + 1}..."}, room=client_sid)
                flask_logger.info(f"BG Task {client_sid}: Day {day_number} processed. Proceeding to next day.")
        
    except Exception as loop_exc: # Catch any unhandled exceptions from the while loop itself
        flask_logger.error(f"Critical unhandled exception in simulation loop for SID {client_sid}: {loop_exc}", exc_info=True)
        socketio.emit('simulation_error', {'error': f"A critical server error occurred: {str(loop_exc)}. Simulation stopped.", 'fatal': True}, room=client_sid)
        stop_event.set() # Ensure the loop terminates
    finally:
        # This block executes whether the loop finished normally, by break, or by unhandled exception (if caught by outer try/finally)
        final_message = "Simulation session stopped." 
        if stop_event.is_set():
            final_message = "Simulation session stopped or terminated due to an error or user request."
        if session_had_a_day_where_both_algos_failed:
                final_message += " There was at least one day where both algorithms failed."
        
        flask_logger.info(f"BG Task {client_sid}: Continuous simulation loop FINISHED/TERMINATED for {algo_names_str}, Patient {selected_patient_id_str}. Stop_event: {stop_event.is_set()}.")
        socketio.emit('simulation_finished', {'message': final_message}, room=client_sid)
        
        if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules: del client_meal_schedules[client_sid]

@app.route('/')
def index(): return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    flask_logger.info(f"Client connected: {request.sid}")
    socketio.emit('connection_ack', {'message': 'Connected!'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    client_sid = request.sid
    flask_logger.info(f"Client disconnected: {client_sid}")
    if client_sid in running_simulations_stop_events:
        if running_simulations_stop_events[client_sid]: # Check if event object exists
            running_simulations_stop_events[client_sid].set()
        flask_logger.info(f"Stop event set for SID: {client_sid}")
    if client_sid in client_meal_schedules:
        del client_meal_schedules[client_sid]

@socketio.on('start_continuous_simulation')
def handle_start_continuous_simulation(data):
    client_sid = request.sid
    flask_logger.info(f"Socket 'start_continuous_simulation' from {client_sid} with data: {data}")
    # Check if a simulation is already running for this client
    if client_sid in running_simulations_stop_events and \
       running_simulations_stop_events.get(client_sid) and \
       not running_simulations_stop_events[client_sid].is_set():
        socketio.emit('simulation_error', {"error": "Simulation is already running for this session."}, room=client_sid); return

    try:
        # Validate input data
        if not data or 'meals' not in data or 'comparison_mode' not in data or 'patient_id' not in data:
            socketio.emit('simulation_error', {"error": "Missing 'meals', 'comparison_mode', or 'patient_id' data.", 'fatal':True}, room=client_sid); return

        selected_comparison_mode = data.get('comparison_mode')
        selected_patient_id_str = data.get('patient_id')
        meal_details_input = data.get('meals', [])

        if selected_comparison_mode not in VALID_COMPARISON_MODES:
            socketio.emit('simulation_error', {"error": f"Invalid comparison mode: {selected_comparison_mode}.", 'fatal':True}, room=client_sid); return
        if selected_patient_id_str not in VALID_PATIENT_IDS:
            socketio.emit('simulation_error', {"error": f"Invalid patient ID: {selected_patient_id_str}.", 'fatal':True}, room=client_sid); return

        algo_pair = selected_comparison_mode.split('_vs_')
        if len(algo_pair) != 2 or not all(algo in ALGORITHMS for algo in algo_pair):
            socketio.emit('simulation_error', {"error": f"Malformed comparison mode: {selected_comparison_mode}.", 'fatal':True}, room=client_sid); return
        algorithms_to_compare_list = algo_pair

        parsed_meal_data_for_loop = []
        if not isinstance(meal_details_input, list):
            socketio.emit('simulation_error', {"error": "'meals' must be a list.", 'fatal':True}, room=client_sid); return
        for meal_item in meal_details_input:
            if not isinstance(meal_item, dict) or "time_minutes" not in meal_item or "carbs" not in meal_item:
                socketio.emit('simulation_error', {"error": "Meal item format incorrect. Expected {time_minutes: X, carbs: Y}.", 'fatal':True}, room=client_sid); return
            try:
                time_min = int(meal_item["time_minutes"]); carbs = float(meal_item["carbs"])
                if time_min < 0 or carbs <= 0: # Basic validation
                    socketio.emit('simulation_error', {"error": "Meal time must be non-negative and carbs must be positive.", 'fatal':True}, room=client_sid); return
                if not (0 <= time_min < SIM_DURATION_MINUTES): # Ensure meal time is within a day
                    socketio.emit('simulation_error', {"error": f"Meal time {time_min} must be within 0 and {SIM_DURATION_MINUTES-1} minutes.", 'fatal':True}, room=client_sid); return
                parsed_meal_data_for_loop.append({'time_minutes': time_min, 'carbs': carbs})
            except ValueError:
                socketio.emit('simulation_error', {"error": "Invalid number format in meal data.", 'fatal':True}, room=client_sid); return
        
        if not parsed_meal_data_for_loop: # Log if no meals, but don't make it fatal
             flask_logger.info(f"No meals input from UI for SID {client_sid}. Sim will run with default (0g) meals as per scenario config if not overridden by patient files.")

        # Store meal schedule and set up stop event
        client_meal_schedules[client_sid] = parsed_meal_data_for_loop
        stop_event = threading.Event()
        running_simulations_stop_events[client_sid] = stop_event

        algo_names_display = " & ".join(algo.upper() for algo in algorithms_to_compare_list)
        socketio.emit('simulation_starting_continuous', {"message": f"Simulation initializing to compare {algo_names_display} for Patient {selected_patient_id_str}..."}, room=client_sid)
        # Clear any previous visualization state on the client
        socketio.emit('clear_visualization', room=client_sid) 

        # Start the background task
        socketio.start_background_task(
            target=run_continuous_simulation_loop,
            client_sid=client_sid,
            meal_data_list_of_dicts=parsed_meal_data_for_loop,
            stop_event=stop_event,
            algorithms_to_compare_list=algorithms_to_compare_list,
            selected_patient_id_str=selected_patient_id_str
        )
        flask_logger.info(f"Started BG task for continuous sim for SID: {client_sid} comparing {algo_names_display}, Patient: {selected_patient_id_str}")
    except Exception as e: # Catch any exceptions during the setup of the simulation start
        flask_logger.error(f"Error in 'start_continuous_simulation' for SID {client_sid}: {e}", exc_info=True)
        socketio.emit('simulation_error', {"error": f"Server error starting simulation: {str(e)}", 'fatal':True}, room=client_sid)
        # Clean up if setup fails
        if client_sid in running_simulations_stop_events:
            if running_simulations_stop_events.get(client_sid): # Check if event object exists
                running_simulations_stop_events[client_sid].set()
            del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules:
            del client_meal_schedules[client_sid]

@socketio.on('stop_continuous_simulation')
def handle_stop_continuous_simulation():
    client_sid = request.sid
    flask_logger.info(f"Socket 'stop_continuous_simulation' from {client_sid}")
    if client_sid in running_simulations_stop_events:
        stop_event = running_simulations_stop_events[client_sid]
        if stop_event and not stop_event.is_set(): # Check if event object exists and not already set
            stop_event.set()
            flask_logger.info(f"Stop signal sent to BG task for SID: {client_sid}")
            socketio.emit('simulation_stopping_ack', {'message': 'Stop signal received by server.'}, room=client_sid)
        else:
            flask_logger.info(f"Simulation for SID {client_sid} was already stopping or stopped, or event was None.")
            socketio.emit('simulation_stopping_ack', {'message': 'Simulation already stopping/stopped.'}, room=client_sid)
    else:
        socketio.emit('simulation_error', {'error': 'No active simulation found to stop for this session.'}, room=client_sid)

if __name__ == '__main__':
    project_root_for_env = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(project_root_for_env, '.env')
    if not os.path.exists(env_file_path):
        with open(env_file_path, 'w') as f: f.write(f"MAIN_PATH={project_root_for_env}\n")
        flask_logger.info(f"Created .env: {env_file_path}")
    if 'MAIN_PATH' not in os.environ:
         os.environ['MAIN_PATH'] = project_root_for_env
         flask_logger.info(f"MAIN_PATH set from project root (fallback): {os.environ['MAIN_PATH']}")
    try:
        from decouple import config as decouple_config
        main_path_val_env = decouple_config('MAIN_PATH', default=os.environ['MAIN_PATH'])
        os.environ['MAIN_PATH'] = main_path_val_env
        flask_logger.info(f"MAIN_PATH set/confirmed: {main_path_val_env}")
    except Exception as e:
        flask_logger.error(f"Error with decouple for MAIN_PATH: {e}. MAIN_PATH: {os.environ.get('MAIN_PATH')}")
        if 'MAIN_PATH' not in os.environ: # Should not be needed if above fallback works, but defensive
            os.environ['MAIN_PATH'] = project_root_for_env
            flask_logger.warning(f"MAIN_PATH re-set to project root: {project_root_for_env}")

    app.config['VERBOSE_DEBUG'] = os.environ.get('FLASK_VERBOSE_DEBUG', 'False').lower() == 'true'
    flask_logger.info("Starting Flask-SocketIO server.")
    socketio.run(app,
                 debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
                 use_reloader=os.environ.get('FLASK_USE_RELOADER', 'False').lower() == 'true',
                 host='0.0.0.0',
                 port=int(os.environ.get("PORT", 5001))
                )