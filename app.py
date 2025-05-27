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

from agents.models.actor_critic import ActorCritic # Assuming this is your model class
from utils.worker import OnPolicyWorker # Assuming this is your worker class

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_very_secret_key_for_socketio_forever_v9!' # Updated
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=50)


flask_logger = app.logger
if not flask_logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')
    flask_logger = logging.getLogger(__name__)


MODEL_DIR = "./models/"
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}"
PCPO_MODEL_FILENAME_BASE = "pcpo_model_for_patient_{}"
SWITCHING_MODEL_FILENAME_BASE = "switching_model_for_patient_{}"

MODEL_FILENAME_BASES = {
    "ppo": PPO_MODEL_FILENAME_BASE,
    "pcpo": PCPO_MODEL_FILENAME_BASE,
    "switching": SWITCHING_MODEL_FILENAME_BASE
}

SIM_DURATION_MINUTES = 24 * 60
SIM_SAMPLING_RATE = 5
MIN_ALIVE_STEPS_THRESHOLD = 5 # If an algo runs for fewer than this many steps in a segment, it's considered "dead" for future segments.

VALID_PATIENT_IDS = [str(i) for i in range(9)] + [str(i) for i in range(20, 30)]
VALID_COMPARISON_MODES = ["ppo_vs_pcpo", "ppo_vs_switching", "pcpo_vs_switching"]
ALGORITHMS = ["ppo", "pcpo", "switching"] # For validation

running_simulations_stop_events = {}
client_meal_schedules = {}


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_simulation_args(run_identifier_prefix="sim_run", patient_id_str="0"):
    args = Namespace()
    unique_log_id = f"{run_identifier_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id)
    args.experiment_dir = args.experiment_folder
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', False)
    current_seed = random.randint(0, 1000000)
    args.seed = current_seed

    project_root_for_env_util = os.path.dirname(os.path.abspath(__file__))
    main_path_val_util = os.environ.get('MAIN_PATH', project_root_for_env_util)
    original_main_path = os.environ.get('MAIN_PATH')
    if not original_main_path:
        os.environ['MAIN_PATH'] = main_path_val_util

    from environment.utils import get_patient_env # Ensure this import works
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
    except ValueError as e:
        flask_logger.error(f"Error resolving patient_id_str '{patient_id_str}': {e}. Defaulting to patient_id 0.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "patient#000"
        args.patient_name = "0"
    except IndexError:
        flask_logger.error(f"IndexError for patient_id {args.patient_id} (from '{patient_id_str}'). Defaulting.")
        args.patient_id = 0
        args.patient_name_for_env = "patient#000"
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
            flask_logger.warning(f"Meal entry {i+1} (time: {custom_meal_times_hours[i]}h) exceeds {num_meal_slots} scenario meal slots. Ignoring.")

    worker_args.meal_times_mean = cfg_meal_times_mean
    worker_args.val_meal_amount = cfg_val_meal_amount
    worker_args.val_meal_prob = cfg_val_meal_prob
    worker_args.val_time_variance = [1e-8] * num_meal_slots
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(cfg_meal_times_mean)
    worker_args.time_upper_bound = list(cfg_meal_times_mean)
    worker_args.env_type = 'testing'
    return worker_args

def run_simulation_segment(client_sid, model_filename_base, meal_data_tuples, agent_type_tag, sim_id_segment, stop_event, patient_name_str_override):
    flask_logger.info(f"Enter run_simulation_segment for {agent_type_tag}, patient {patient_name_str_override}, segment ID: {sim_id_segment}")
    actual_model_filename_partial = model_filename_base.format(patient_name_str_override)
    base_args = get_simulation_args(run_identifier_prefix=sim_id_segment, patient_id_str=patient_name_str_override)
    worker_args = configure_custom_scenario_args(base_args, meal_data_tuples)

    original_main_path_temp_env = os.environ.get('MAIN_PATH')
    if not original_main_path_temp_env: os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
    from environment.utils import get_patient_env
    patients_list_full, _ = get_patient_env()
    if not original_main_path_temp_env: del os.environ['MAIN_PATH']
    elif original_main_path_temp_env != os.environ.get('MAIN_PATH'): os.environ['MAIN_PATH'] = original_main_path_temp_env

    if not (0 <= worker_args.patient_id < len(patients_list_full)):
        err_msg = f"Invalid patient_id {worker_args.patient_id} for {agent_type_tag}."
        flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag}, room=client_sid); return {"error": err_msg}
    worker_args.patient_name_for_env = patients_list_full[worker_args.patient_id]

    actor_path = os.path.join(MODEL_DIR, f"patient{patient_name_str_override}", f"{actual_model_filename_partial}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"patient{patient_name_str_override}", f"{actual_model_filename_partial}_Critic.pth")

    if not os.path.exists(actor_path):
        err_msg = f"Actor model not found: {actor_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag}, room=client_sid); return {"error": err_msg}
    if not os.path.exists(critic_path):
        err_msg = f"Critic model not found: {critic_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag}, room=client_sid); return {"error": err_msg}

    sim_results = {}; original_main_path_worker_ctx = os.environ.get('MAIN_PATH')
    if not original_main_path_worker_ctx: os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))

    try:
        policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
        policy_net.to(worker_args.device); policy_net.eval()
        api_worker_id = abs(hash(sim_id_segment + agent_type_tag + str(time.time()))) % 10000 + 7000
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id)
        worker.rollout(policy=policy_net, buffer=None)
        if stop_event.is_set(): return None
        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        if not os.path.exists(log_file_path):
            err_msg = f"Log NOT FOUND: {log_file_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag}, room=client_sid); return {"error": err_msg}
        df = pd.read_csv(log_file_path)
        if df.empty:
            flask_logger.warning(f"Log IS EMPTY: {log_file_path} for {agent_type_tag}. Sim likely terminated at step 0.")
            return {'cgm': [], 'insulin': [], 'meal': [], 'rewards': [], 'patient_name': worker_args.patient_name, 'duration_steps': 0}

        required_cols = ['cgm', 'ins', 'rew', 'meals_input_per_step']
        for col in required_cols:
            if col not in df.columns:
                if col == 'meals_input_per_step': df[col] = 0.0
                else: err_msg = f"Missing column '{col}' in {log_file_path}"; flask_logger.error(err_msg); socketio.emit('simulation_error', {'error': err_msg, 'agent_type_tag': agent_type_tag}, room=client_sid); return {"error": err_msg}
        if df[['cgm', 'ins', 'rew']].isnull().any().any(): flask_logger.warning(f"NaNs in critical columns in {log_file_path}.")
        sim_results = {'cgm': df['cgm'].tolist(), 'insulin': df['ins'].tolist(), 'meal': df['meals_input_per_step'].tolist(), 'rewards': df['rew'].tolist(), 'patient_name': worker_args.patient_name, 'duration_steps': len(df)}
    except Exception as e:
        flask_logger.error(f"EXCEPTION in run_simulation_segment for {agent_type_tag}: {e}", exc_info=True)
        return {"error": f"Exception in {agent_type_tag} segment: {str(e)}"}
    finally:
        if not original_main_path_worker_ctx and 'MAIN_PATH' in os.environ: del os.environ['MAIN_PATH']
        elif original_main_path_worker_ctx and os.environ.get('MAIN_PATH') != original_main_path_worker_ctx: os.environ['MAIN_PATH'] = original_main_path_worker_ctx
        folder_to_cleanup = getattr(base_args, 'experiment_folder', None)
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try: shutil.rmtree(folder_to_cleanup)
            except Exception as e_clean: flask_logger.error(f"Error cleaning temp dir {folder_to_cleanup}: {e_clean}")
    return sim_results

def emit_meal_markers_for_view(client_sid, current_segment_start_step, steps_in_day_view):
    if client_sid not in client_meal_schedules: return
    meal_schedule_for_client = client_meal_schedules[client_sid]
    markers = []
    for seg_num in range( (current_segment_start_step // steps_in_day_view) -1 , (current_segment_start_step // steps_in_day_view) + 2 ):
        if seg_num < 0: continue
        segment_base_step = seg_num * steps_in_day_view
        for meal_info_dict in meal_schedule_for_client:
            meal_time_min = meal_info_dict['time_minutes']
            meal_carbs = meal_info_dict['carbs']
            meal_step_within_segment = meal_time_min // SIM_SAMPLING_RATE
            global_meal_step = segment_base_step + meal_step_within_segment
            markers.append({'step': global_meal_step, 'carbs': meal_carbs})
    socketio.emit('meal_markers', {'markers': markers}, room=client_sid)

# ... (imports and other functions remain the same) ...

def run_continuous_simulation_loop(client_sid, meal_data_list_of_dicts, stop_event, algorithms_to_compare_list, selected_patient_id_str):
    algo_names_str = " & ".join(algo.upper() for algo in algorithms_to_compare_list)
    flask_logger.info(f"BG Task for {client_sid}: Comparing {algo_names_str}, Patient: {selected_patient_id_str}.")
    segment_count = 0
    global_step_offset = 0
    client_meal_schedules[client_sid] = meal_data_list_of_dicts

    algo1_name = algorithms_to_compare_list[0]
    algo2_name = algorithms_to_compare_list[1]
    
    algo1_is_alive = True 
    algo2_is_alive = True

    try:
        temp_args_for_name = get_simulation_args(f"init_name_{client_sid}", patient_id_str=selected_patient_id_str)
        patient_name_for_sim_display = temp_args_for_name.patient_name
        initial_sampling_rate = temp_args_for_name.sampling_rate
        steps_per_day_segment = SIM_DURATION_MINUTES // initial_sampling_rate
        del temp_args_for_name
    except Exception as e: 
        flask_logger.error(f"BG Task {client_sid}: Error determining patient name/rate: {e}", exc_info=True)
        socketio.emit('simulation_error', {'error': 'Init failed determining patient info.'}, room=client_sid)
        if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules: del client_meal_schedules[client_sid]
        return

    socketio.emit('simulation_metadata', {
        'patient_name': patient_name_for_sim_display,
        'algorithms_compared': algorithms_to_compare_list,
        'sampling_rate': initial_sampling_rate
    }, room=client_sid)

    emit_meal_markers_for_view(client_sid, 0, steps_per_day_segment)
    meal_data_tuples_for_worker = [(m['time_minutes'], m['carbs']) for m in meal_data_list_of_dicts]
    
    algo1_model_base = MODEL_FILENAME_BASES[algo1_name]
    algo2_model_base = MODEL_FILENAME_BASES[algo2_name]

    while not stop_event.is_set():
        if not algo1_is_alive and not algo2_is_alive:
            flask_logger.info(f"BG Task {client_sid}: Both algorithms ({algo1_name.upper()}, {algo2_name.upper()}) are permanently stopped. Ending continuous simulation.")
            socketio.emit('simulation_error', {'error': f"Both {algo1_name.upper()} and {algo2_name.upper()} simulations terminated previously and will not restart."}, room=client_sid)
            break 
            
        segment_count += 1
        # **** ADDED DETAILED LOGGING HERE ****
        flask_logger.info(f"BG Task {client_sid}: SEGMENT {segment_count} PROCESSING START. Patient: {selected_patient_id_str}")
        flask_logger.info(f"                   Algo1 ({algo1_name}) Status: {'ALIVE' if algo1_is_alive else 'DEAD'}")
        flask_logger.info(f"                   Algo2 ({algo2_name}) Status: {'ALIVE' if algo2_is_alive else 'DEAD'}")
        # ************************************

        sim_id_segment_base = f"sid_{client_sid}_comp_{algo1_name}-{algo2_name}_p_{selected_patient_id_str}_seg_{segment_count}"

        algo1_data_this_segment = {'duration_steps': 0, 'cgm': [], 'insulin': [], 'rewards': [], 'meal': []} 
        algo2_data_this_segment = {'duration_steps': 0, 'cgm': [], 'insulin': [], 'rewards': [], 'meal': []}
        
        duration1_this_segment = 0
        duration2_this_segment = 0

        if algo1_is_alive:
            socketio.emit('simulation_progress', {'message': f'Computing {algo1_name.upper()} for P{selected_patient_id_str}, seg {segment_count}...'}, room=client_sid)
            algo1_start_time = time.time()
            temp_data = run_simulation_segment(client_sid, algo1_model_base, meal_data_tuples_for_worker, algo1_name, f"{sim_id_segment_base}_{algo1_name}", stop_event, selected_patient_id_str)
            
            if isinstance(temp_data, dict) and "error" not in temp_data:
                algo1_data_this_segment = temp_data
                duration1_this_segment = algo1_data_this_segment.get('duration_steps', 0)
                if duration1_this_segment <= MIN_ALIVE_STEPS_THRESHOLD:
                    flask_logger.warning(f"{algo1_name.upper()} died in seg {segment_count} ({duration1_this_segment} steps). Marking permanently dead."); 
                    algo1_is_alive = False 
            elif isinstance(temp_data, dict) and "error" in temp_data:
                 flask_logger.error(f"Error in {algo1_name} seg {segment_count}: {temp_data.get('error', 'Unknown')}. Marking permanently dead."); 
                 algo1_is_alive = False; algo1_data_this_segment['error_message'] = temp_data.get('error')
            elif temp_data is None and stop_event.is_set(): 
                 flask_logger.info(f"Stop event during {algo1_name} run for seg {segment_count}."); break
            else: 
                 flask_logger.error(f"Unexpected return from {algo1_name} seg {segment_count}. Marking permanently dead."); 
                 algo1_is_alive = False; algo1_data_this_segment['error_message'] = 'Unknown error or None returned'
            flask_logger.info(f"BG Task {client_sid}: {algo1_name.upper()} seg {segment_count} took {time.time() - algo1_start_time:.2f}s. Duration: {duration1_this_segment}. Now alive: {algo1_is_alive}")
        else:
            flask_logger.info(f"BG Task {client_sid}: {algo1_name.upper()} is already dead. Skipping run for seg {segment_count}.")

        if stop_event.is_set(): break

        if algo2_is_alive:
            socketio.emit('simulation_progress', {'message': f'Computing {algo2_name.upper()} for P{selected_patient_id_str}, seg {segment_count}...'}, room=client_sid)
            algo2_start_time = time.time()
            temp_data = run_simulation_segment(client_sid, algo2_model_base, meal_data_tuples_for_worker, algo2_name, f"{sim_id_segment_base}_{algo2_name}", stop_event, selected_patient_id_str)

            if isinstance(temp_data, dict) and "error" not in temp_data:
                algo2_data_this_segment = temp_data
                duration2_this_segment = algo2_data_this_segment.get('duration_steps', 0)
                if duration2_this_segment <= MIN_ALIVE_STEPS_THRESHOLD:
                    flask_logger.warning(f"{algo2_name.upper()} died in seg {segment_count} ({duration2_this_segment} steps). Marking permanently dead."); 
                    algo2_is_alive = False
            elif isinstance(temp_data, dict) and "error" in temp_data:
                 flask_logger.error(f"Error in {algo2_name} seg {segment_count}: {temp_data.get('error', 'Unknown')}. Marking permanently dead.");
                 algo2_is_alive = False; algo2_data_this_segment['error_message'] = temp_data.get('error')
            elif temp_data is None and stop_event.is_set():
                 flask_logger.info(f"Stop event during {algo2_name} run for seg {segment_count}."); break
            else:
                 flask_logger.error(f"Unexpected return from {algo2_name} seg {segment_count}. Marking permanently dead.");
                 algo2_is_alive = False; algo2_data_this_segment['error_message'] = 'Unknown error or None returned'
            flask_logger.info(f"BG Task {client_sid}: {algo2_name.upper()} seg {segment_count} took {time.time() - algo2_start_time:.2f}s. Duration: {duration2_this_segment}. Now alive: {algo2_is_alive}")
        else: 
            flask_logger.info(f"BG Task {client_sid}: {algo2_name.upper()} is already dead. Skipping run for seg {segment_count}.")

        if stop_event.is_set(): break
        
        num_steps_to_stream = max(duration1_this_segment, duration2_this_segment)
        
        emit_meal_markers_for_view(client_sid, global_step_offset, steps_per_day_segment)
        if num_steps_to_stream > 0 :
            socketio.emit('simulation_progress', {'message': f'Segment {segment_count} results. Streaming {num_steps_to_stream} steps...'}, room=client_sid)
        elif not algo1_is_alive and not algo2_is_alive:
             socketio.emit('simulation_progress', {'message': f'Segment {segment_count}: Both algorithms stopped.'}, room=client_sid)
        else: 
             socketio.emit('simulation_progress', {'message': f'Segment {segment_count}: No new data to stream this segment.'}, room=client_sid)

        for i in range(num_steps_to_stream):
            if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Stop during streaming seg {segment_count} step {i}."); break
            current_global_step = global_step_offset + i
            try:
                data_point = {'step': current_global_step, 'algo1_tag': algo1_name, 'algo2_tag': algo2_name}
                
                # Algo1 data
                # Only send actual data if algo1 was alive at the START of this segment's processing AND current step is within its run for THIS segment
                if algo1_is_alive or i < duration1_this_segment : # This logic was a bit off, should be simpler
                    if i < duration1_this_segment : # Data exists for this step from algo1 for this current segment's run
                        data_point.update({
                            'algo1_cgm': algo1_data_this_segment['cgm'][i], 
                            'algo1_insulin': algo1_data_this_segment['insulin'][i], 
                            'algo1_reward': algo1_data_this_segment['rewards'][i],
                        })
                        data_point['meal'] = algo1_data_this_segment['meal'][i] # Prioritize meal from algo1 if it ran this step
                    else: # Algo1 was alive at start but died before this step i in this segment
                        data_point.update({'algo1_cgm': None, 'algo1_insulin': None, 'algo1_reward': None})
                else: # Algo1 was already dead before this segment began (algo1_is_alive was false)
                    data_point.update({'algo1_cgm': None, 'algo1_insulin': None, 'algo1_reward': None})
                
                # Algo2 data
                if algo2_is_alive or i < duration2_this_segment:
                    if i < duration2_this_segment:
                        data_point.update({
                            'algo2_cgm': algo2_data_this_segment['cgm'][i], 
                            'algo2_insulin': algo2_data_this_segment['insulin'][i], 
                            'algo2_reward': algo2_data_this_segment['rewards'][i]
                        })
                        if data_point.get('meal', None) is None: # If meal not set by algo1
                             data_point['meal'] = algo2_data_this_segment['meal'][i]
                    else:
                        data_point.update({'algo2_cgm': None, 'algo2_insulin': None, 'algo2_reward': None})
                else:
                    data_point.update({'algo2_cgm': None, 'algo2_insulin': None, 'algo2_reward': None})

                if data_point.get('meal', None) is None: # Ensure meal key is always present
                    data_point['meal'] = 0

                socketio.emit('simulation_data_point', data_point, room=client_sid)
                socketio.sleep(0.005)
            except IndexError: 
                flask_logger.error(f"IndexError streaming. Global:{current_global_step}, Local:{i}, Seg:{segment_count}. Dur1:{duration1_this_segment}, Dur2:{duration2_this_segment}", exc_info=True)
                socketio.emit('simulation_error', {'error': 'Internal error: Data streaming inconsistency.'}, room=client_sid)
                stop_event.set(); break 
            except Exception as e_stream:
                flask_logger.error(f"Error emitting data. Global:{current_global_step}: {e_stream}", exc_info=True)
                break

        if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Loop ending post-streaming seg {segment_count} due to stop."); break
        global_step_offset += num_steps_to_stream
        
    flask_logger.info(f"BG Task {client_sid}: Continuous simulation loop finished for {algo_names_str}, Patient {selected_patient_id_str}. Stop_event: {stop_event.is_set()}. Algo1 Final Alive: {algo1_is_alive}, Algo2 Final Alive: {algo2_is_alive}")
    socketio.emit('simulation_finished', {'message': 'Simulation stopped or completed.'}, room=client_sid)
    if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
    if client_sid in client_meal_schedules: del client_meal_schedules[client_sid]

# ... (rest of app.py is identical to the previous full version)
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
        running_simulations_stop_events[client_sid].set()
        flask_logger.info(f"Stop event set for SID: {client_sid}")
    if client_sid in client_meal_schedules:
        del client_meal_schedules[client_sid]

@socketio.on('start_continuous_simulation')
def handle_start_continuous_simulation(data):
    client_sid = request.sid
    flask_logger.info(f"Socket 'start_continuous_simulation' from {client_sid} with data: {data}")
    if client_sid in running_simulations_stop_events and not running_simulations_stop_events[client_sid].is_set():
        socketio.emit('simulation_error', {"error": "Simulation is already running for this session."}, room=client_sid); return

    try:
        if not data or 'meals' not in data or 'comparison_mode' not in data or 'patient_id' not in data:
            socketio.emit('simulation_error', {"error": "Missing 'meals', 'comparison_mode', or 'patient_id' data."}, room=client_sid); return

        selected_comparison_mode = data.get('comparison_mode')
        selected_patient_id_str = data.get('patient_id')
        meal_details_input = data.get('meals', [])

        if selected_comparison_mode not in VALID_COMPARISON_MODES:
            socketio.emit('simulation_error', {"error": f"Invalid comparison mode: {selected_comparison_mode}."}, room=client_sid); return
        if selected_patient_id_str not in VALID_PATIENT_IDS:
            socketio.emit('simulation_error', {"error": f"Invalid patient ID: {selected_patient_id_str}."}, room=client_sid); return

        algo_pair = selected_comparison_mode.split('_vs_')
        if len(algo_pair) != 2 or not all(algo in ALGORITHMS for algo in algo_pair):
            socketio.emit('simulation_error', {"error": f"Malformed comparison mode: {selected_comparison_mode}."}, room=client_sid); return
        algorithms_to_compare_list = algo_pair

        parsed_meal_data_for_loop = []
        if not isinstance(meal_details_input, list):
            socketio.emit('simulation_error', {"error": "'meals' must be a list."}, room=client_sid); return
        for meal_item in meal_details_input:
            if not isinstance(meal_item, dict) or "time_minutes" not in meal_item or "carbs" not in meal_item:
                socketio.emit('simulation_error', {"error": "Meal item format incorrect. Expected {time_minutes: X, carbs: Y}."}, room=client_sid); return
            try:
                time_min = int(meal_item["time_minutes"]); carbs = float(meal_item["carbs"])
                if time_min < 0 or carbs <= 0: 
                    socketio.emit('simulation_error', {"error": "Meal time must be non-negative and carbs must be positive."}, room=client_sid); return
                parsed_meal_data_for_loop.append({'time_minutes': time_min, 'carbs': carbs})
            except ValueError:
                socketio.emit('simulation_error', {"error": "Invalid number format in meal data."}, room=client_sid); return
        
        if not parsed_meal_data_for_loop:
             flask_logger.info(f"No meals input from UI for SID {client_sid}. Sim will run with default (0g) meals defined by scenario config.")

        client_meal_schedules[client_sid] = parsed_meal_data_for_loop
        stop_event = threading.Event()
        running_simulations_stop_events[client_sid] = stop_event
        algo_names_display = " & ".join(algo.upper() for algo in algorithms_to_compare_list)
        socketio.emit('simulation_starting_continuous', {"message": f"Sim initializing to compare {algo_names_display} for Patient {selected_patient_id_str}..."}, room=client_sid)

        socketio.start_background_task(
            target=run_continuous_simulation_loop,
            client_sid=client_sid,
            meal_data_list_of_dicts=parsed_meal_data_for_loop,
            stop_event=stop_event,
            algorithms_to_compare_list=algorithms_to_compare_list,
            selected_patient_id_str=selected_patient_id_str
        )
        flask_logger.info(f"Started BG task for continuous sim for SID: {client_sid} comparing {algo_names_display}, Patient: {selected_patient_id_str}")
    except Exception as e:
        flask_logger.error(f"Error in 'start_continuous_simulation' for SID {client_sid}: {e}", exc_info=True)
        socketio.emit('simulation_error', {"error": f"Server error starting simulation: {str(e)}"}, room=client_sid)
        if client_sid in running_simulations_stop_events:
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
        if not stop_event.is_set():
            stop_event.set()
            flask_logger.info(f"Stop signal sent to BG task for SID: {client_sid}")
            socketio.emit('simulation_stopping_ack', {'message': 'Stop signal received by server.'}, room=client_sid)
        else:
            flask_logger.info(f"Simulation for SID {client_sid} was already stopping or stopped.")
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
        if 'MAIN_PATH' not in os.environ:
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