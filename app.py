import os
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template # Keep jsonify for potential HTTP endpoints
from flask_socketio import SocketIO, emit # emit is also fine in handlers, but socketio.emit is safer from threads
from argparse import Namespace
from datetime import datetime
import logging
import shutil
import uuid
import time 
import threading 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io

from agents.models.actor_critic import ActorCritic
from utils.worker import OnPolicyWorker


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_very_secret_key_for_socketio_forever_v2!' # Change if needed
# Explicitly use eventlet or gevent if installed for better performance with background tasks
# Otherwise, it defaults to threading, which is what we are using with socketio.start_background_task
# For production, 'eventlet' or 'gevent' are recommended.
# pip install eventlet
# pip install gevent
# async_mode = None # Let Flask-SocketIO choose, or set to 'threading', 'eventlet', 'gevent'
# try:
#     import eventlet
#     async_mode = 'eventlet'
# except ImportError:
#     pass
# if async_mode is None:
#     try:
#         from gevent import pywsgi
#         from geventwebsocket.handler import WebSocketHandler
#         async_mode = 'gevent_websocket'
#     except ImportError:
#         pass
# if async_mode is None:
#     async_mode = 'threading' # Fallback
# app.logger.info(f"Using async_mode: {async_mode}")
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
socketio = SocketIO(app, cors_allowed_origins="*") # Defaults to threading if eventlet/gevent not used explicitly with run

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')

MODEL_DIR = "./models/"
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}"
PCPO_MODEL_FILENAME_BASE = "pcpo_model_for_patient_{}"
SIM_PATIENT_NAME_STR = '0' 
SIM_DURATION_MINUTES = 24 * 60 # Duration of one segment/loop iteration
SIM_SAMPLING_RATE = 5

running_simulations_stop_events = {}


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_simulation_args(run_identifier_prefix="sim_run"):
    args = Namespace()
    unique_log_id = f"{run_identifier_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    # temp_web_sim_output should be cleaned up by run_simulation_segment_and_stream
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id)
    args.experiment_dir = args.experiment_folder # Ensure experiment_dir is also set
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', False)
    current_seed = random.randint(0, 1000000) # Generate a new seed for each segment
    args.seed = current_seed 
    app.logger.debug(f"Using seed {current_seed} for simulation segment args.")


    # MAIN_PATH handling for get_patient_env
    project_root_for_env_util = os.path.dirname(os.path.abspath(__file__))
    # It's critical that MAIN_PATH is correctly set in os.environ BEFORE this import if agent.py uses it.
    # The if __name__ == '__main__' block handles this for the main process.
    # For background threads, ensure os.environ['MAIN_PATH'] is stable or passed.
    main_path_val_util = os.environ.get('MAIN_PATH', project_root_for_env_util) # Get from env or default
    
    # Temporarily ensure MAIN_PATH is set for get_patient_env if it relies on it
    # This is a bit of a workaround for background threads. Better if modules always get MAIN_PATH via a passed arg or robust config.
    original_main_path = os.environ.get('MAIN_PATH')
    if not original_main_path:
        os.environ['MAIN_PATH'] = main_path_val_util

    from environment.utils import get_patient_env
    patients_from_util_with_hash, _ = get_patient_env()

    if not original_main_path: # Restore if we set it temporarily
        del os.environ['MAIN_PATH']
    elif original_main_path != os.environ.get('MAIN_PATH'): # Or if it was changed by get_patient_env and we want to revert
        os.environ['MAIN_PATH'] = original_main_path


    try:
        args.patient_id = int(SIM_PATIENT_NAME_STR)
        if not (0 <= args.patient_id < len(patients_from_util_with_hash)):
            app.logger.warning(f"Patient ID {args.patient_id} out of bounds for {len(patients_from_util_with_hash)} patients. Defaulting to 0.")
            args.patient_id = 0 # Default to 0 if out of bounds
        args.patient_name_for_env = patients_from_util_with_hash[args.patient_id]
        args.patient_name = str(args.patient_id) # Use the integer ID as the 'name' for consistency
    except ValueError as e:
        app.logger.error(f"Error resolving SIM_PATIENT_NAME_STR ('{SIM_PATIENT_NAME_STR}'): {e}. Defaulting to patient_id 0.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "patient#000" # Fallback patient name
        args.patient_name = str(args.patient_id)
    except IndexError: # Catch if patients_from_util_with_hash is empty or ID is still bad
        app.logger.error(f"IndexError for patient_id {args.patient_id}. Defaulting.")
        args.patient_id = 0
        args.patient_name_for_env = "patient#000" # Hardcoded fallback
        args.patient_name = "0"


    args.n_action = 1; args.n_features = 2; args.feature_history = 12; args.calibration = 12
    args.control_space_type = "exponential"; args.insulin_min = 0.0; args.insulin_max = 5.0
    args.glucose_max = 600.0; args.glucose_min = 39.0; args.sensor = "GuardianRT"; args.pump = "Insulet"
    args.t_meal = 20; args.sampling_rate = SIM_SAMPLING_RATE
    args.use_meal_announcement = False; args.use_carb_announcement = False; args.use_tod_announcement = False
    args.n_rnn_hidden = 16; args.n_rnn_layers = 1; args.rnn_directions = 1; args.bidirectional = False
    args.return_type = 'average'; args.gamma = 0.99; args.lambda_ = 0.95; args.normalize_reward = True
    args.n_step = SIM_DURATION_MINUTES // SIM_SAMPLING_RATE 
    args.max_epi_length = args.n_step; args.max_test_epi_len = args.n_step
    args.debug = False; args.pi_lr = 3e-4; args.vf_lr = 3e-4
    num_meal_slots = 6 
    args.meal_times_mean = [8.0, 10.5, 13.0, 16.5, 20.0, 22.5][:num_meal_slots] # Hours
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

    # Initialize with potentially different defaults if base_args could change them
    # For val_ scenario, we usually want fixed meals based on input.
    new_meal_times_mean = [0.0] * num_meal_slots 
    new_val_meal_amount = [0.0] * num_meal_slots
    new_val_meal_prob = [-1.0] * num_meal_slots # Default to no meal unless specified

    for i in range(len(custom_meal_times_hours)):
        if i < num_meal_slots:
            new_meal_times_mean[i] = custom_meal_times_hours[i]
            new_val_meal_amount[i] = custom_meal_amounts_grams[i]
            new_val_meal_prob[i] = 1.0 # Meal occurs
        else:
            logging.warning(f"Meal entry {i+1} (time: {custom_meal_times_hours[i]}h, carbs: {custom_meal_amounts_grams[i]}g) exceeds available scenario meal slots ({num_meal_slots}). Ignoring.")

    # These are for the validation/testing scenario specifically
    worker_args.meal_times_mean = new_meal_times_mean # This will be used by RandomScenario if env_type='testing'
    worker_args.val_meal_amount = new_val_meal_amount
    worker_args.val_meal_prob = new_val_meal_prob
    
    worker_args.val_time_variance = [1e-8] * num_meal_slots
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(new_meal_times_mean) 
    worker_args.time_upper_bound = list(new_meal_times_mean) 
    
    worker_args.env_type = 'testing' # CRITICAL: This tells RandomScenario to use val_ params
    return worker_args


def run_continuous_simulation_loop(client_sid, meal_data, stop_event):
    app.logger.info(f"BG Task for {client_sid}: Continuous simulation loop started.")
    segment_count = 0
    global_step_offset = 0 

    # Determine patient_name once (assuming it doesn't change during the continuous run)
    # This is a bit of a hack; ideally, get_simulation_args shouldn't rely on global SIM_PATIENT_NAME_STR
    # if it's meant to be dynamic or configurable per request/session.
    try:
        temp_args_for_name = get_simulation_args(f"init_name_{client_sid}")
        patient_name_for_sim = temp_args_for_name.patient_name
        initial_sampling_rate = temp_args_for_name.sampling_rate
        del temp_args_for_name
    except Exception as e:
        app.logger.error(f"BG Task for {client_sid}: Could not determine patient name/sampling rate. Error: {e}")
        socketio.emit('simulation_error', {'error': 'Failed to initialize simulation parameters.'}, room=client_sid)
        return

    socketio.emit('simulation_metadata', {
        'patient_name': patient_name_for_sim,
        'sampling_rate': initial_sampling_rate
    }, room=client_sid)


    while not stop_event.is_set():
        segment_count += 1
        app.logger.info(f"BG Task for {client_sid}: Starting segment {segment_count}.")
        # Create a unique sim_id for this specific segment for logging/temp files
        sim_id_segment_base = f"sid_{client_sid}_seg_{segment_count}"

        # --- PPO Agent Segment ---
        if not stop_event.is_set():
            app.logger.info(f"BG Task for {client_sid}: Running PPO segment {segment_count}")
            ppo_segment_results = run_simulation_segment_and_stream(
                client_sid, PPO_MODEL_FILENAME_BASE, meal_data, "ppo", 
                f"{sim_id_segment_base}_ppo", stop_event, global_step_offset, patient_name_for_sim
            )
            if ppo_segment_results and "error" in ppo_segment_results:
                app.logger.error(f"BG Task for {client_sid}: PPO Error in segment {segment_count}: {ppo_segment_results['error']}")
                socketio.emit('simulation_error', {'agent': 'ppo', 'error': ppo_segment_results['error']}, room=client_sid)
            elif ppo_segment_results is None: 
                app.logger.info(f"BG Task for {client_sid}: PPO segment {segment_count} interrupted by stop signal.")
                break


        # --- PCPO Agent Segment ---
        if not stop_event.is_set():
            app.logger.info(f"BG Task for {client_sid}: Running PCPO segment {segment_count}")
            pcpo_segment_results = run_simulation_segment_and_stream(
                client_sid, PCPO_MODEL_FILENAME_BASE, meal_data, "pcpo", 
                f"{sim_id_segment_base}_pcpo", stop_event, global_step_offset, patient_name_for_sim
            )
            if pcpo_segment_results and "error" in pcpo_segment_results:
                app.logger.error(f"BG Task for {client_sid}: PCPO Error in segment {segment_count}: {pcpo_segment_results['error']}")
                socketio.emit('simulation_error', {'agent': 'pcpo', 'error': pcpo_segment_results['error']}, room=client_sid)
            elif pcpo_segment_results is None: 
                app.logger.info(f"BG Task for {client_sid}: PCPO segment {segment_count} interrupted by stop signal.")
                break
        
        if stop_event.is_set():
            app.logger.info(f"BG Task for {client_sid}: Stop signal detected after agent segments. Ending loop.")
            break

        current_segment_duration_steps = SIM_DURATION_MINUTES // initial_sampling_rate
        global_step_offset += current_segment_duration_steps

        socketio.emit('segment_complete', {'segment_number': segment_count, 'steps_in_segment': current_segment_duration_steps}, room=client_sid)
        app.logger.info(f"BG Task for {client_sid}: Segment {segment_count} fully processed.")
        socketio.sleep(0.5) # Small pause between full day segments

    app.logger.info(f"BG Task for {client_sid}: Continuous simulation loop finished.")
    socketio.emit('simulation_finished', {'message': 'Simulation stopped by user or completed.'}, room=client_sid)
    if client_sid in running_simulations_stop_events:
        del running_simulations_stop_events[client_sid]


def run_simulation_segment_and_stream(client_sid, model_template, meal_data, agent_type, sim_id_segment, stop_event, global_step_offset, patient_name_override):
    actual_model_filename = model_template.format(patient_name_override) # Use the consistent patient name
    
    # Pass a unique identifier for this segment run to get_simulation_args
    base_args = get_simulation_args(run_identifier_prefix=sim_id_segment) 
    # Override patient_name in base_args if needed, though get_simulation_args should now use SIM_PATIENT_NAME_STR
    # base_args.patient_name = patient_name_override 
    # base_args.patient_id = int(patient_name_override) # Assuming SIM_PATIENT_NAME_STR is just the ID

    worker_args = configure_custom_scenario_args(base_args, meal_data)
    # Ensure the correct patient is set for the worker_args if not already handled by get_simulation_args
    worker_args.patient_name = patient_name_override 
    worker_args.patient_id = int(patient_name_override) # Ensure ID is int for get_patient_env lookup if needed
    # Find the corresponding patient_name_for_env
    original_main_path_temp = os.environ.get('MAIN_PATH')
    if not original_main_path_temp: os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
    from environment.utils import get_patient_env
    patients_list_full, _ = get_patient_env()
    if not original_main_path_temp: del os.environ['MAIN_PATH']
    
    if 0 <= worker_args.patient_id < len(patients_list_full):
        worker_args.patient_name_for_env = patients_list_full[worker_args.patient_id]
    else:
        app.logger.error(f"Invalid patient_id {worker_args.patient_id} for {agent_type} in segment {sim_id_segment}.")
        return {"error": f"Configuration error for patient ID for {agent_type}."}


    actor_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Critic.pth")

    if not os.path.exists(actor_path): return {"error": f"Actor model for {agent_type} (patient {patient_name_override}) not found: {actor_path}"}
    if not os.path.exists(critic_path): return {"error": f"Critic model for {agent_type} (patient {patient_name_override}) not found: {critic_path}"}

    sim_results_full_segment = {}
    # Ensure MAIN_PATH is set for OnPolicyWorker context
    original_main_path = os.environ.get('MAIN_PATH')
    if not original_main_path:
        os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__)) # Default if not set
        app.logger.info(f"Temporarily setting MAIN_PATH for worker: {os.environ['MAIN_PATH']}")

    try:
        policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
        policy_net.to(worker_args.device); policy_net.eval()

        # Ensure worker_id is unique for each call to avoid log file overwrites if not cleaned up fast enough
        api_worker_id = abs(hash(sim_id_segment + agent_type + str(time.time()))) % 10000 + 7000 
        
        app.logger.info(f"Streaming {agent_type} for SID {client_sid}, segment_id {sim_id_segment}, worker_id {api_worker_id}, patient {worker_args.patient_name_for_env}")
        
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id)
        worker.rollout(policy=policy_net, buffer=None) 

        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        if not os.path.exists(log_file_path):
            app.logger.error(f"Sim log for {agent_type} NOT FOUND: {log_file_path}")
            return {"error": f"Sim log for {agent_type} not found: {log_file_path}"}

        df = pd.read_csv(log_file_path)
        if df.empty:
            app.logger.error(f"Sim log for {agent_type} IS EMPTY: {log_file_path}")
            return {"error": f"Sim log for {agent_type} is empty."}
        
        required_cols = ['cgm', 'ins', 'rew']
        for col in required_cols:
            if col not in df.columns:
                app.logger.error(f"Missing required column '{col}' in log file: {log_file_path} for agent {agent_type}")
                return {"error": f"Missing data column '{col}' for {agent_type}."}

        if 'meals_input_per_step' not in df.columns: 
            app.logger.warning(f"Column 'meals_input_per_step' not in {log_file_path}, defaulting to 0 for agent {agent_type}.")
            df['meals_input_per_step'] = 0.0

        for index, row in df.iterrows():
            if stop_event.is_set():
                app.logger.info(f"Stop event detected during streaming for {agent_type}, SID {client_sid}.")
                return None 
            data_point = {
                'agent': agent_type,
                'step': global_step_offset + index, 
                'cgm': row['cgm'],
                'insulin': row['ins'],
                'meal': row.get('meals_input_per_step', 0.0), 
                'reward': row['rew']
            }
            socketio.emit('simulation_data_point', data_point, room=client_sid)
            socketio.sleep(0.01) # Adjusted sleep time for smoother streaming

        sim_results_full_segment = {
            'cgm': df['cgm'].tolist(), 'insulin': df['ins'].tolist(),
            'meal': df['meals_input_per_step'].tolist(), 'rewards': df['rew'].tolist(),
            'patient_name': worker_args.patient_name, 'duration_steps': len(df)
        }

    except Exception as e:
        app.logger.error(f"Exception during {agent_type} segment for SID {client_sid} (sim_id {sim_id_segment}): {e}", exc_info=True)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': f"Exception: {str(e)}"}, room=client_sid)
        return {"error": f"Exception in {agent_type} segment: {str(e)}"}
    finally:
        if not original_main_path and 'MAIN_PATH' in os.environ: # Clean up MAIN_PATH if we set it
            del os.environ['MAIN_PATH']
            app.logger.info(f"Restored MAIN_PATH after worker execution for {agent_type}.")
        elif original_main_path and os.environ.get('MAIN_PATH') != original_main_path: # Or if it was changed and we want to revert
             os.environ['MAIN_PATH'] = original_main_path


        folder_to_cleanup = getattr(base_args, 'experiment_folder', None)
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try:
                shutil.rmtree(folder_to_cleanup)
                app.logger.info(f"Cleaned up temp dir: {folder_to_cleanup}")
            except Exception as e:
                app.logger.error(f"Error cleaning temp dir {folder_to_cleanup}: {e}")
    return sim_results_full_segment


# plot_simulation_results_to_file can remain largely the same.
# It's optional for saving snapshots and not directly part of the live streaming.
# ... (plot_simulation_results_to_file function from previous correct answer)

@app.route('/')
def index():
    return render_template('index.html')

# --- Socket.IO event handlers ---
@socketio.on('connect')
def handle_connect():
    app.logger.info(f"Client connected: {request.sid}")
    socketio.emit('connection_ack', {'message': 'Connected to simulation server!'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    client_sid = request.sid
    app.logger.info(f"Client disconnected: {client_sid}")
    if client_sid in running_simulations_stop_events:
        running_simulations_stop_events[client_sid].set() 
        app.logger.info(f"Stop event set for simulation associated with disconnecting SID: {client_sid}")
        # The background task will clean up from the dictionary upon exiting

@socketio.on('start_continuous_simulation')
def handle_start_continuous_simulation(data):
    client_sid = request.sid
    app.logger.info(f"Socket 'start_continuous_simulation' from {client_sid} with data: {data}")

    if client_sid in running_simulations_stop_events and \
       not running_simulations_stop_events[client_sid].is_set():
        socketio.emit('simulation_error', {"error": "A simulation is already running for your session. Please stop it first."}, room=client_sid)
        return

    try:
        if not data or 'meals' not in data: # Check if data and 'meals' key exist
            socketio.emit('simulation_error', {"error": "Missing 'meals' data in request."}, room=client_sid)
            return
        
        meal_details_input = data.get('meals', []) # Safely get meals, defaults to []
        parsed_meal_data = []
        if not isinstance(meal_details_input, list):
            socketio.emit('simulation_error', {"error": "'meals' data must be a list."}, room=client_sid)
            return

        for meal_item in meal_details_input:
            if not isinstance(meal_item, dict) or "time_minutes" not in meal_item or "carbs" not in meal_item:
                socketio.emit('simulation_error', {"error": "Each meal item must be an object with 'time_minutes' and 'carbs'."}, room=client_sid)
                return
            try:
                time_min = int(meal_item["time_minutes"])
                carbs = float(meal_item["carbs"])
                if time_min < 0 or carbs < 0: # Basic validation
                    socketio.emit('simulation_error', {"error": "Meal time and carbs cannot be negative."}, room=client_sid)
                    return
                parsed_meal_data.append((time_min, carbs))
            except ValueError:
                socketio.emit('simulation_error', {"error": "Invalid number format for meal time or carbs."}, room=client_sid)
                return
        parsed_meal_data.sort(key=lambda x: x[0]) # Sort meals by time

        stop_event = threading.Event()
        running_simulations_stop_events[client_sid] = stop_event
        
        socketio.emit('simulation_starting_continuous', {"message": "Continuous simulation initializing..."}, room=client_sid)
        
        socketio.start_background_task(
            target=run_continuous_simulation_loop,
            client_sid=client_sid,
            meal_data=parsed_meal_data,
            stop_event=stop_event
        )
        app.logger.info(f"Started background task for continuous simulation for SID: {client_sid}")

    except Exception as e:
        app.logger.error(f"Error in 'start_continuous_simulation' handler for SID {client_sid}: {e}", exc_info=True)
        socketio.emit('simulation_error', {"error": f"Server error starting simulation: {str(e)}"}, room=client_sid)
        if client_sid in running_simulations_stop_events: 
            running_simulations_stop_events[client_sid].set() 
            del running_simulations_stop_events[client_sid]


@socketio.on('stop_continuous_simulation')
def handle_stop_continuous_simulation():
    client_sid = request.sid
    app.logger.info(f"Socket 'stop_continuous_simulation' received from {client_sid}")
    if client_sid in running_simulations_stop_events:
        stop_event = running_simulations_stop_events[client_sid]
        if not stop_event.is_set():
            stop_event.set()
            app.logger.info(f"Stop signal sent to background task for SID: {client_sid}")
            socketio.emit('simulation_stopping_ack', {'message': 'Stop signal received. Simulation will end.'}, room=client_sid)
        else:
            app.logger.info(f"Simulation for SID {client_sid} was already stopping.")
            socketio.emit('simulation_stopping_ack', {'message': 'Simulation already stopping or stopped.'}, room=client_sid)
    else:
        socketio.emit('simulation_error', {'error': 'No active simulation found to stop for your session.'}, room=client_sid)


if __name__ == '__main__':
    project_root_for_env = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(project_root_for_env, '.env')

    if not os.path.exists(env_file_path):
        with open(env_file_path, 'w') as f:
            f.write(f"MAIN_PATH={project_root_for_env}\n")
        logging.info(f"Created .env file at {env_file_path} with MAIN_PATH={project_root_for_env}")
    
    from decouple import config as decouple_config
    try:
        # It's crucial that MAIN_PATH is loaded and set before other modules needing it are imported by workers/threads
        main_path_val = decouple_config('MAIN_PATH', default=project_root_for_env)
        os.environ['MAIN_PATH'] = main_path_val 
        logging.info(f"MAIN_PATH set to: {main_path_val} (from .env or default)")
    except Exception as e: 
        logging.error(f"CRITICAL: Error setting MAIN_PATH using decouple: {e}. Defaulting to project root.")
        os.environ['MAIN_PATH'] = project_root_for_env
        logging.info(f"MAIN_PATH set to (fallback): {project_root_for_env}")
    
    app.config['VERBOSE_DEBUG'] = True # For app's own debug logic
    
    # For development with Flask's reloader, use_reloader=True might be needed.
    # However, allow_unsafe_werkzeug is generally for eventlet/gevent with reloader.
    # If using default threading async_mode, `debug=True` alone for socketio.run might be enough.
    # If `WERKZEUG_SERVER_FD` error persists, try `use_reloader=False` first with `debug=True`.
    socketio.run(app, 
                 debug=True, # Enables Flask debug mode and Werkzeug reloader by default
                 host='0.0.0.0', 
                 port=int(os.environ.get("PORT", 5001))
                 # Forcing reloader: use_reloader=True
                 # If using eventlet/gevent AND reloader: allow_unsafe_werkzeug=True
                )