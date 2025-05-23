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

from agents.models.actor_critic import ActorCritic
from utils.worker import OnPolicyWorker

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_very_secret_key_for_socketio_forever_v2!'
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=50)


flask_logger = app.logger
if not flask_logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')
    flask_logger = logging.getLogger(__name__)


MODEL_DIR = "./models/"
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}"
PCPO_MODEL_FILENAME_BASE = "pcpo_model_for_patient_{}"
SIM_PATIENT_NAME_STR = '0'
SIM_DURATION_MINUTES = 24 * 60
SIM_SAMPLING_RATE = 5 # minutes per step

running_simulations_stop_events = {}
# Store meal schedules per client session
client_meal_schedules = {}


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ... get_simulation_args and configure_custom_scenario_args remain the same ...
def get_simulation_args(run_identifier_prefix="sim_run"):
    args = Namespace()
    unique_log_id = f"{run_identifier_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id)
    args.experiment_dir = args.experiment_folder
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data") # Used by worker
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', True) 
    current_seed = random.randint(0, 1000000)
    args.seed = current_seed # This seed is for the overall segment args
    flask_logger.debug(f"Using seed {current_seed} for simulation segment args for {run_identifier_prefix}.")

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
        args.patient_id = int(SIM_PATIENT_NAME_STR)
        if not (0 <= args.patient_id < len(patients_from_util_with_hash)):
            flask_logger.warning(f"Patient ID {args.patient_id} out of bounds. Defaulting to 0.")
            args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[args.patient_id]
        args.patient_name = str(args.patient_id)
    except ValueError as e:
        flask_logger.error(f"Error resolving SIM_PATIENT_NAME_STR: {e}. Defaulting to patient_id 0.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "patient#000"
        args.patient_name = str(args.patient_id)
    except IndexError:
        flask_logger.error(f"IndexError for patient_id {args.patient_id}. Defaulting.")
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
    args.meal_times_mean = [8.0, 10.5, 13.0, 16.5, 20.0, 22.5][:num_meal_slots]
    args.time_variance = [1e-8] * num_meal_slots; args.time_lower_bound = list(args.meal_times_mean)
    args.time_upper_bound = list(args.meal_times_mean); args.meal_prob = [-1.0] * num_meal_slots
    args.meal_amount = [0.0] * num_meal_slots; args.meal_variance = [1e-8] * num_meal_slots
    args.val_meal_prob = [-1.0] * num_meal_slots; args.val_meal_amount = [0.0] * num_meal_slots
    args.val_meal_variance = [1e-8] * num_meal_slots; args.val_time_variance = [1e-8] * num_meal_slots
    return args

def configure_custom_scenario_args(base_args, scenario_meal_data):
    worker_args = Namespace(**vars(base_args)) 
    custom_meal_times_hours = []
    custom_meal_amounts_grams = []
    if scenario_meal_data:
        for meal_time_min, meal_carbs in scenario_meal_data:
            custom_meal_times_hours.append(float(meal_time_min) / 60.0)
            custom_meal_amounts_grams.append(float(meal_carbs))

    num_meal_slots = len(worker_args.meal_times_mean) 
    new_meal_times_mean = [0.0] * num_meal_slots 
    new_val_meal_amount = [0.0] * num_meal_slots
    new_val_meal_prob = [-1.0] * num_meal_slots 

    for i in range(len(custom_meal_times_hours)):
        if i < num_meal_slots:
            new_meal_times_mean[i] = custom_meal_times_hours[i]
            new_val_meal_amount[i] = custom_meal_amounts_grams[i]
            new_val_meal_prob[i] = 1.0 
        else:
            flask_logger.warning(f"Meal entry {i+1} (time: {custom_meal_times_hours[i]}h) exceeds available scenario meal slots ({num_meal_slots}). Ignoring.")
 
    worker_args.meal_times_mean = new_meal_times_mean 
    worker_args.val_meal_amount = new_val_meal_amount
    worker_args.val_meal_prob = new_val_meal_prob
    
    worker_args.val_time_variance = [1e-8] * num_meal_slots 
    worker_args.val_meal_variance = [1e-8] * num_meal_slots 
    worker_args.time_lower_bound = list(new_meal_times_mean) 
    worker_args.time_upper_bound = list(new_meal_times_mean) 
    
    worker_args.env_type = 'testing' 
    return worker_args

# ... run_simulation_segment remains the same as the last version ...
def run_simulation_segment(client_sid, model_template, meal_data, agent_type, sim_id_segment, stop_event, patient_name_override):
    flask_logger.info(f"Enter run_simulation_segment for {agent_type}, segment ID: {sim_id_segment}")
    actual_model_filename = model_template.format(patient_name_override)
    base_args = get_simulation_args(run_identifier_prefix=sim_id_segment) 
    worker_args = configure_custom_scenario_args(base_args, meal_data) 
    worker_args.patient_name = patient_name_override 
    worker_args.patient_id = int(patient_name_override) 


    original_main_path_temp_env = os.environ.get('MAIN_PATH')
    if not original_main_path_temp_env:
        os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
    from environment.utils import get_patient_env 
    patients_list_full, _ = get_patient_env()
    if not original_main_path_temp_env:
        del os.environ['MAIN_PATH']
    elif original_main_path_temp_env != os.environ.get('MAIN_PATH'):
        os.environ['MAIN_PATH'] = original_main_path_temp_env

    if 0 <= worker_args.patient_id < len(patients_list_full):
        worker_args.patient_name_for_env = patients_list_full[worker_args.patient_id] 
    else:
        err_msg = f"Invalid patient_id {worker_args.patient_id} for {agent_type}."
        flask_logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}

    actor_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Critic.pth")

    if not os.path.exists(actor_path):
        err_msg = f"Actor model not found: {actor_path}"
        flask_logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}
    if not os.path.exists(critic_path):
        err_msg = f"Critic model not found: {critic_path}"
        flask_logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}

    sim_results = {}
    original_main_path_worker_ctx = os.environ.get('MAIN_PATH')
    if not original_main_path_worker_ctx:
        os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
        flask_logger.info(f"Temporarily setting MAIN_PATH for {agent_type} worker: {os.environ['MAIN_PATH']}")

    try:
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Loading policy. Actor: {actor_path}, Critic: {critic_path}")
        policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
        policy_net.to(worker_args.device); policy_net.eval()
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Policy net loaded.")

        api_worker_id = abs(hash(sim_id_segment + agent_type + str(time.time()))) % 10000 + 7000 
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] OnPolicyWorker: id {api_worker_id}, patient {worker_args.patient_name_for_env}")
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id) 
        
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Calling worker.rollout() for {api_worker_id}. BLOCKING.")
        worker.rollout(policy=policy_net, buffer=None) 
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] worker.rollout() COMPLETED for {api_worker_id}.")

        if stop_event.is_set():
            flask_logger.warning(f"[{sim_id_segment} | {agent_type}] Stop event IS SET after worker.rollout().")
            return None

        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv") 
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Attempting to read log: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            err_msg = f"Log NOT FOUND after rollout: {log_file_path}"
            flask_logger.error(err_msg)
            socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
            return {"error": err_msg}
            
        df = pd.read_csv(log_file_path)
        flask_logger.info(f"Log {log_file_path} for {agent_type} read. Shape: {df.shape}")

        if df.empty:
            err_msg = f"Log IS EMPTY: {log_file_path}"
            flask_logger.error(err_msg)
            socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
            return {"error": err_msg}

        required_cols = ['cgm', 'ins', 'rew', 'meals_input_per_step'] 
        for col in required_cols:
            if col not in df.columns:
                if col == 'meals_input_per_step':
                    flask_logger.warning(f"Column '{col}' not found in {log_file_path} for {agent_type}. Defaulting to 0.")
                    df[col] = 0.0 
                else:
                    err_msg = f"Missing required column '{col}' in {log_file_path}"
                    flask_logger.error(err_msg)
                    socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
                    return {"error": err_msg}


        if df[['cgm', 'ins', 'rew']].isnull().any().any(): 
            flask_logger.warning(f"NaNs in critical cgm/ins/rew columns for {agent_type} in {log_file_path}.")
        
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Processed log. Steps: {len(df)}")
        sim_results = {
            'cgm': df['cgm'].tolist(), 'insulin': df['ins'].tolist(),
            'meal': df['meals_input_per_step'].tolist(), 
            'rewards': df['rew'].tolist(),
            'patient_name': worker_args.patient_name, 'duration_steps': len(df)
        }
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Returning sim_results.")

    except Exception as e:
        flask_logger.error(f"CRITICAL EXCEPTION in run_simulation_segment for {agent_type} (SID {client_sid}, sim_id {sim_id_segment}): {e}", exc_info=True)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': f"Backend exception: {str(e)}"}, room=client_sid)
        return {"error": f"Exception in {agent_type} segment: {str(e)}"}
    finally:
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Finally block for cleanup.")
        if not original_main_path_worker_ctx and 'MAIN_PATH' in os.environ:
            del os.environ['MAIN_PATH']
            flask_logger.info(f"Restored MAIN_PATH after {agent_type} worker.")
        elif original_main_path_worker_ctx and os.environ.get('MAIN_PATH') != original_main_path_worker_ctx:
             os.environ['MAIN_PATH'] = original_main_path_worker_ctx
        folder_to_cleanup = getattr(base_args, 'experiment_folder', None)
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try:
                shutil.rmtree(folder_to_cleanup)
                flask_logger.info(f"Cleaned temp dir: {folder_to_cleanup}")
            except Exception as e_clean:
                flask_logger.error(f"Error cleaning temp dir {folder_to_cleanup}: {e_clean}")
        flask_logger.info(f"[{sim_id_segment} | {agent_type}] Exiting run_simulation_segment.")
    return sim_results


def emit_meal_markers_for_view(client_sid, current_segment_start_step, steps_in_day_view):
    """
    Calculates and emits meal markers (plotLines) for the current viewing window.
    """
    if client_sid not in client_meal_schedules:
        return

    meal_schedule_for_client = client_meal_schedules[client_sid] # List of (time_min, carbs)
    markers = []
    
    # The view window on the frontend is from current_segment_start_step to current_segment_start_step + steps_in_day_view -1
    # However, meal_schedule_for_client times are relative to the start of *a* 24-hour segment (0 to 1440 mins)
    
    for seg_num in range( (current_segment_start_step // steps_in_day_view) -1 , (current_segment_start_step // steps_in_day_view) + 2 ):
        # Consider meals from previous, current, and next logical "day" segment
        # to catch meals that might fall into the sliding window
        if seg_num < 0: continue

        segment_base_step = seg_num * steps_in_day_view

        for meal_time_min, meal_carbs in meal_schedule_for_client:
            meal_step_within_segment = meal_time_min // SIM_SAMPLING_RATE
            global_meal_step = segment_base_step + meal_step_within_segment

            # Check if this global_meal_step falls within the current frontend view
            # view_min_step = current_segment_start_step
            # view_max_step = current_segment_start_step + steps_in_day_view -1
            # A simpler way for plotLines: Highcharts will only draw them if their value is within the xAxis extremes.
            # So we can send all potentially relevant ones around the current view.
            markers.append({
                'step': global_meal_step,
                'carbs': meal_carbs
            })
            
    flask_logger.debug(f"Emitting meal markers for SID {client_sid}: {markers}")
    socketio.emit('meal_markers', {'markers': markers}, room=client_sid)


def run_continuous_simulation_loop(client_sid, meal_data_from_frontend, stop_event): # Renamed meal_data
    flask_logger.info(f"BG Task for {client_sid}: Continuous simulation loop started.")
    segment_count = 0
    global_step_offset = 0 # This is the start step of the current segment being processed

    # Store the original meal schedule (time_in_min, carbs) for this client
    # This is used by emit_meal_markers_for_view
    client_meal_schedules[client_sid] = meal_data_from_frontend 

    try:
        temp_args_for_name = get_simulation_args(f"init_name_{client_sid}")
        patient_name_for_sim = temp_args_for_name.patient_name
        initial_sampling_rate = temp_args_for_name.sampling_rate
        steps_per_day_segment = SIM_DURATION_MINUTES // initial_sampling_rate
        del temp_args_for_name
    except Exception as e:
        flask_logger.error(f"BG Task {client_sid}: Error determining patient name/rate: {e}", exc_info=True)
        socketio.emit('simulation_error', {'error': 'Init failed.'}, room=client_sid)
        if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules: del client_meal_schedules[client_sid]
        return

    socketio.emit('simulation_metadata', {
        'patient_name': patient_name_for_sim, 'sampling_rate': initial_sampling_rate
    }, room=client_sid)
    
    # Emit initial meal markers for the first view (steps 0 to steps_per_day_segment-1)
    emit_meal_markers_for_view(client_sid, 0, steps_per_day_segment)


    while not stop_event.is_set():
        segment_count += 1
        flask_logger.info(f"BG Task {client_sid}: ============== STARTING SEGMENT {segment_count} ==============")
        flask_logger.info(f"BG Task {client_sid}: Global_step_offset for new segment: {global_step_offset}")
        sim_id_segment_base = f"sid_{client_sid}_seg_{segment_count}"

        # configure_custom_scenario_args uses meal_data_from_frontend to set up the scenario for the worker
        # This means each worker effectively gets the same daily meal pattern for its 24h run
        
        socketio.emit('simulation_progress', {'message': f'Computing PPO for segment {segment_count}...'}, room=client_sid)
        ppo_start_time = time.time()
        ppo_segment_data = run_simulation_segment(client_sid, PPO_MODEL_FILENAME_BASE, meal_data_from_frontend, "ppo", f"{sim_id_segment_base}_ppo", stop_event, patient_name_for_sim)
        ppo_duration = time.time() - ppo_start_time
        flask_logger.info(f"BG Task {client_sid}: PPO for seg {segment_count} took {ppo_duration:.2f}s")
        if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Stop after PPO segment. Ending."); break
        if ppo_segment_data and "error" in ppo_segment_data: flask_logger.error(f"BG Task {client_sid}: PPO Error seg {segment_count}: {ppo_segment_data['error']}"); break
        if ppo_segment_data is None: flask_logger.info(f"BG Task {client_sid}: PPO seg {segment_count} returned None. Ending."); break

        socketio.emit('simulation_progress', {'message': f'PPO segment {segment_count} done. Computing PCPO...'}, room=client_sid)
        pcpo_start_time = time.time()
        pcpo_segment_data = run_simulation_segment(client_sid, PCPO_MODEL_FILENAME_BASE, meal_data_from_frontend, "pcpo", f"{sim_id_segment_base}_pcpo", stop_event, patient_name_for_sim)
        pcpo_duration = time.time() - pcpo_start_time
        flask_logger.info(f"BG Task {client_sid}: PCPO for seg {segment_count} took {pcpo_duration:.2f}s")
        if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Stop after PCPO segment. Ending."); break
        if pcpo_segment_data and "error" in pcpo_segment_data: flask_logger.error(f"BG Task {client_sid}: PCPO Error seg {segment_count}: {pcpo_segment_data['error']}"); break
        if pcpo_segment_data is None: flask_logger.info(f"BG Task {client_sid}: PCPO seg {segment_count} returned None. Ending."); break

        socketio.emit('simulation_progress', {'message': f'PCPO segment {segment_count} done. Streaming data...'}, room=client_sid)
        if not isinstance(ppo_segment_data, dict) or not isinstance(pcpo_segment_data, dict) or \
           'duration_steps' not in ppo_segment_data or 'duration_steps' not in pcpo_segment_data:
            err_msg = f"Invalid segment data structure for seg {segment_count}."
            flask_logger.error(f"BG Task {client_sid}: {err_msg}")
            socketio.emit('simulation_error', {'error': err_msg}, room=client_sid); break
        
        num_steps_in_segment_data = ppo_segment_data.get('duration_steps', 0) # Actual steps returned by worker
        if num_steps_in_segment_data == 0 or num_steps_in_segment_data != pcpo_segment_data.get('duration_steps', -1):
            err_msg = f"Data length mismatch/zero for seg {segment_count}. PPO: {num_steps_in_segment_data}, PCPO: {pcpo_segment_data.get('duration_steps', -1)}"
            flask_logger.error(f"BG Task {client_sid}: {err_msg}")
            socketio.emit('simulation_error', {'error': err_msg}, room=client_sid); break
        
        # Emit meal markers for the upcoming view window
        # The view window starts at global_step_offset
        emit_meal_markers_for_view(client_sid, global_step_offset, steps_per_day_segment)

        flask_logger.info(f"BG Task {client_sid}: Streaming data for seg {segment_count} ({num_steps_in_segment_data} steps). Offset: {global_step_offset}")
        for i in range(num_steps_in_segment_data):
            if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Stop during streaming seg {segment_count} local step {i}."); break
            current_global_step = global_step_offset + i
            
            # ... (verbose logging for specific steps - can be kept or removed) ...

            try:
                combined_data_point = {
                    'step': current_global_step,
                    'ppo_cgm': ppo_segment_data['cgm'][i], 'ppo_insulin': ppo_segment_data['insulin'][i], 'ppo_reward': ppo_segment_data['rewards'][i],
                    'pcpo_cgm': pcpo_segment_data['cgm'][i], 'pcpo_insulin': pcpo_segment_data['insulin'][i], 'pcpo_reward': pcpo_segment_data['rewards'][i],
                    'meal': ppo_segment_data['meal'][i], # This meal is from worker's log, should be 0 if handled by plotLines
                }
                socketio.emit('simulation_data_point', combined_data_point, room=client_sid)
                socketio.sleep(0.005) 
            except IndexError: # ... (error handling) ...
                flask_logger.error(f"IndexError creating combined_data_point. Global: {current_global_step}, Local: {i}, Seg: {segment_count}.", exc_info=True)
                socketio.emit('simulation_error', {'error': 'Internal error: Inconsistent data.'}, room=client_sid)
                stop_event.set(); break
            except Exception as e_stream: # ... (error handling) ...
                flask_logger.error(f"Error emitting data. Global: {current_global_step}: {e_stream}", exc_info=True)
                break 
        
        if stop_event.is_set(): flask_logger.info(f"BG Task {client_sid}: Loop ending post-streaming seg {segment_count} due to stop."); break
        global_step_offset += num_steps_in_segment_data # Use actual steps from data
        flask_logger.info(f"BG Task {client_sid}: Seg {segment_count} processed. New offset: {global_step_offset}")

    flask_logger.info(f"BG Task {client_sid}: Continuous simulation loop finished. Stop_event: {stop_event.is_set()}")
    socketio.emit('simulation_finished', {'message': 'Simulation stopped or completed.'}, room=client_sid)
    if client_sid in running_simulations_stop_events: del running_simulations_stop_events[client_sid]
    if client_sid in client_meal_schedules: del client_meal_schedules[client_sid] # Clean up meal schedule

# ... (Flask routes and SocketIO handlers for connect, disconnect, start, stop remain the same) ...
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
    if client_sid in client_meal_schedules: # Clean up meal schedule on disconnect
        del client_meal_schedules[client_sid]


@socketio.on('start_continuous_simulation')
def handle_start_continuous_simulation(data):
    client_sid = request.sid
    flask_logger.info(f"Socket 'start_continuous_simulation' from {client_sid} with data: {data}")
    if client_sid in running_simulations_stop_events and not running_simulations_stop_events[client_sid].is_set():
        socketio.emit('simulation_error', {"error": "Sim already running."}, room=client_sid); return
    try:
        if not data or 'meals' not in data: socketio.emit('simulation_error', {"error": "Missing 'meals' data."}, room=client_sid); return
        meal_details_input = data.get('meals', []) # This is [(time_min, carbs), ...]
        parsed_meal_data_for_worker_config = [] # This is for configure_custom_scenario_args
        
        # Store the original {time_minutes: X, carbs: Y} structure for emit_meal_markers_for_view
        # but configure_custom_scenario_args expects [(time_min, carbs)]
        original_meal_schedule_for_markers = [] 

        if not isinstance(meal_details_input, list): socketio.emit('simulation_error', {"error": "'meals' must be a list."}, room=client_sid); return
        for meal_item in meal_details_input: # meal_item is {time_minutes: X, carbs: Y}
            if not isinstance(meal_item, dict) or "time_minutes" not in meal_item or "carbs" not in meal_item:
                socketio.emit('simulation_error', {"error": "Meal item format incorrect."}, room=client_sid); return
            try:
                time_min = int(meal_item["time_minutes"]); carbs = float(meal_item["carbs"])
                if time_min < 0 or carbs < 0: socketio.emit('simulation_error', {"error": "Meal values non-negative."}, room=client_sid); return
                parsed_meal_data_for_worker_config.append((time_min, carbs)) # For worker scenario
                original_meal_schedule_for_markers.append({'time_minutes': time_min, 'carbs': carbs}) # For backend plotLine logic
            except ValueError: socketio.emit('simulation_error', {"error": "Invalid meal number format."}, room=client_sid); return
        
        # Sort the one used for configuring worker scenario (if necessary, though order might not matter there)
        parsed_meal_data_for_worker_config.sort(key=lambda x: x[0]) 
        
        # Store the original structure with keys for emit_meal_markers_for_view
        client_meal_schedules[client_sid] = original_meal_schedule_for_markers


        stop_event = threading.Event()
        running_simulations_stop_events[client_sid] = stop_event
        socketio.emit('simulation_starting_continuous', {"message": "Sim initializing..."}, room=client_sid)
        # Pass the (time_min, carbs) tuples list to the loop, which then passes to configure_custom_scenario_args
        socketio.start_background_task(target=run_continuous_simulation_loop, client_sid=client_sid, meal_data_from_frontend=parsed_meal_data_for_worker_config, stop_event=stop_event)
        flask_logger.info(f"Started BG task for continuous sim for SID: {client_sid}")
    except Exception as e:
        flask_logger.error(f"Error in 'start_continuous_simulation' for SID {client_sid}: {e}", exc_info=True)
        socketio.emit('simulation_error', {"error": f"Server error starting sim: {str(e)}"}, room=client_sid)
        if client_sid in running_simulations_stop_events:
            running_simulations_stop_events[client_sid].set()
            del running_simulations_stop_events[client_sid]
        if client_sid in client_meal_schedules: # Clean up meal schedule on error
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
            socketio.emit('simulation_stopping_ack', {'message': 'Stop signal received.'}, room=client_sid)
        else:
            flask_logger.info(f"Sim for SID {client_sid} already stopping/stopped.")
            socketio.emit('simulation_stopping_ack', {'message': 'Sim already stopping.'}, room=client_sid)
    else:
        socketio.emit('simulation_error', {'error': 'No active sim to stop.'}, room=client_sid)


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
    
    app.config['VERBOSE_DEBUG'] = False 

    flask_logger.info("Starting Flask-SocketIO server with debug=False, use_reloader=False.")
    socketio.run(app,
                 debug=False, 
                 use_reloader=False, 
                 host='0.0.0.0',
                 port=int(os.environ.get("PORT", 5001))
                )