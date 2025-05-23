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
# import uuid # Not used in provided code, can be removed if still unused
import time
import threading

# import matplotlib # Not used for live streaming in the proposed solution
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import io

from agents.models.actor_critic import ActorCritic
from utils.worker import OnPolicyWorker


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your_very_secret_key_for_socketio_forever_v2!'
socketio = SocketIO(app, cors_allowed_origins="*")

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
    # temp_web_sim_output should be cleaned up by run_simulation_segment
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id)
    args.experiment_dir = args.experiment_folder # Ensure experiment_dir is also set
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', False)
    current_seed = random.randint(0, 1000000) # Generate a new seed for each segment
    args.seed = current_seed
    app.logger.debug(f"Using seed {current_seed} for simulation segment args for {run_identifier_prefix}.")


    # MAIN_PATH handling for get_patient_env
    project_root_for_env_util = os.path.dirname(os.path.abspath(__file__))
    main_path_val_util = os.environ.get('MAIN_PATH', project_root_for_env_util) # Get from env or default

    original_main_path = os.environ.get('MAIN_PATH')
    if not original_main_path:
        os.environ['MAIN_PATH'] = main_path_val_util

    from environment.utils import get_patient_env # This import might depend on MAIN_PATH
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

    worker_args.meal_times_mean = new_meal_times_mean
    worker_args.val_meal_amount = new_val_meal_amount
    worker_args.val_meal_prob = new_val_meal_prob

    worker_args.val_time_variance = [1e-8] * num_meal_slots
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(new_meal_times_mean)
    worker_args.time_upper_bound = list(new_meal_times_mean)

    worker_args.env_type = 'testing' # CRITICAL: This tells RandomScenario to use val_ params
    return worker_args

# Renamed from run_simulation_segment_and_stream, and streaming logic moved out
def run_simulation_segment(client_sid, model_template, meal_data, agent_type, sim_id_segment, stop_event, patient_name_override):
    actual_model_filename = model_template.format(patient_name_override)

    base_args = get_simulation_args(run_identifier_prefix=sim_id_segment)
    worker_args = configure_custom_scenario_args(base_args, meal_data)
    worker_args.patient_name = patient_name_override
    worker_args.patient_id = int(patient_name_override)

    # Ensure MAIN_PATH is set for get_patient_env, then restore original if necessary
    original_main_path_temp_env = os.environ.get('MAIN_PATH')
    if not original_main_path_temp_env:
        os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))

    from environment.utils import get_patient_env # This import might depend on MAIN_PATH
    patients_list_full, _ = get_patient_env()

    if not original_main_path_temp_env: # If MAIN_PATH was not set before, remove our temporary one
        del os.environ['MAIN_PATH']
    elif original_main_path_temp_env != os.environ.get('MAIN_PATH'): # If it was changed by get_patient_env, restore it
        os.environ['MAIN_PATH'] = original_main_path_temp_env

    if 0 <= worker_args.patient_id < len(patients_list_full):
        worker_args.patient_name_for_env = patients_list_full[worker_args.patient_id]
    else:
        err_msg = f"Invalid patient_id {worker_args.patient_id} for {agent_type} in segment {sim_id_segment}."
        app.logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}

    actor_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Critic.pth")

    if not os.path.exists(actor_path):
        err_msg = f"Actor model for {agent_type} (patient {patient_name_override}) not found: {actor_path}"
        app.logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}
    if not os.path.exists(critic_path):
        err_msg = f"Critic model for {agent_type} (patient {patient_name_override}) not found: {critic_path}"
        app.logger.error(err_msg)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
        return {"error": err_msg}

    sim_results = {}
    # Ensure MAIN_PATH is set for OnPolicyWorker context, then restore
    original_main_path_worker_ctx = os.environ.get('MAIN_PATH')
    if not original_main_path_worker_ctx:
        os.environ['MAIN_PATH'] = os.path.dirname(os.path.abspath(__file__))
        app.logger.info(f"Temporarily setting MAIN_PATH for {agent_type} worker: {os.environ['MAIN_PATH']}")

    try:
        policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
        policy_net.to(worker_args.device); policy_net.eval()

        # Ensure worker_id is unique for each call to avoid log file overwrites
        api_worker_id = abs(hash(sim_id_segment + agent_type + str(time.time()))) % 10000 + 7000 # Offset to avoid clashes with training worker IDs

        app.logger.info(f"Running {agent_type} sim for SID {client_sid}, segment_id {sim_id_segment}, worker_id {api_worker_id}, patient {worker_args.patient_name_for_env}")

        # OnPolicyWorker.rollout is blocking. Stop event can only be effectively checked after it.
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id)
        app.logger.debug(f"Calling worker.rollout() for {agent_type}, segment {sim_id_segment}")
        worker.rollout(policy=policy_net, buffer=None) # This call is blocking for the duration of the segment simulation
        app.logger.debug(f"worker.rollout() completed for {agent_type}, segment {sim_id_segment}")


        if stop_event.is_set(): # Check if stop was signaled during the (potentially long) rollout
            app.logger.info(f"Stop event detected after rollout for {agent_type} (SID {client_sid}, segment {sim_id_segment}). Not processing results.")
            return None # Indicate interruption

        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        if not os.path.exists(log_file_path):
            err_msg = f"Sim log for {agent_type} NOT FOUND: {log_file_path}"
            app.logger.error(err_msg)
            socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
            return {"error": err_msg}

        df = pd.read_csv(log_file_path)
        app.logger.info(f"Log file {log_file_path} for {agent_type} read successfully. Shape: {df.shape}")


        if df.empty:
            err_msg = f"Sim log for {agent_type} IS EMPTY: {log_file_path}"
            app.logger.error(err_msg)
            socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
            return {"error": err_msg}

        required_cols = ['cgm', 'ins', 'rew']
        for col in required_cols:
            if col not in df.columns:
                err_msg = f"Missing required column '{col}' in log file: {log_file_path} for agent {agent_type}"
                app.logger.error(err_msg)
                socketio.emit('simulation_error', {'agent': agent_type, 'error': err_msg}, room=client_sid)
                return {"error": err_msg}

        # Check for NaN values which can break things
        if df[required_cols].isnull().any().any():
            app.logger.warning(f"NaN values found in critical columns for {agent_type} in {log_file_path}. This might cause issues.")
            # Consider how to handle NaNs, e.g., df.fillna(0, inplace=True) or more specific handling

        if 'meals_input_per_step' not in df.columns:
            app.logger.warning(f"Column 'meals_input_per_step' not in {log_file_path}, defaulting to 0 for agent {agent_type}.")
            df['meals_input_per_step'] = 0.0

        app.logger.debug(f"Preparing to return data for {agent_type} from segment {sim_id_segment}")
        sim_results = {
            'cgm': df['cgm'].tolist(),
            'insulin': df['ins'].tolist(),
            'meal': df['meals_input_per_step'].tolist(),
            'rewards': df['rew'].tolist(),
            'patient_name': worker_args.patient_name,
            'duration_steps': len(df)
        }

    except Exception as e:
        app.logger.error(f"CRITICAL EXCEPTION in run_simulation_segment for {agent_type} (SID {client_sid}, sim_id {sim_id_segment}): {e}", exc_info=True)
        socketio.emit('simulation_error', {'agent': agent_type, 'error': f"Critical backend exception: {str(e)}"}, room=client_sid)
        return {"error": f"Exception in {agent_type} segment: {str(e)}"}
    finally:
        # Restore MAIN_PATH if it was temporarily set or changed by this function's context
        if not original_main_path_worker_ctx and 'MAIN_PATH' in os.environ:
            del os.environ['MAIN_PATH']
            app.logger.info(f"Restored MAIN_PATH after {agent_type} worker execution.")
        elif original_main_path_worker_ctx and os.environ.get('MAIN_PATH') != original_main_path_worker_ctx:
             os.environ['MAIN_PATH'] = original_main_path_worker_ctx

        # Cleanup temporary directory for this specific segment simulation
        folder_to_cleanup = getattr(base_args, 'experiment_folder', None)
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try:
                shutil.rmtree(folder_to_cleanup)
                app.logger.info(f"Cleaned up temp dir: {folder_to_cleanup}")
            except Exception as e:
                app.logger.error(f"Error cleaning temp dir {folder_to_cleanup}: {e}")
    return sim_results


def run_continuous_simulation_loop(client_sid, meal_data, stop_event):
    app.logger.info(f"BG Task for {client_sid}: Continuous simulation loop started.")
    segment_count = 0
    global_step_offset = 0

    try:
        temp_args_for_name = get_simulation_args(f"init_name_{client_sid}")
        patient_name_for_sim = temp_args_for_name.patient_name
        initial_sampling_rate = temp_args_for_name.sampling_rate # Should be SIM_SAMPLING_RATE
        del temp_args_for_name
    except Exception as e:
        app.logger.error(f"BG Task for {client_sid}: Could not determine patient name/sampling rate. Error: {e}", exc_info=True)
        socketio.emit('simulation_error', {'error': 'Failed to initialize simulation parameters.'}, room=client_sid)
        return

    socketio.emit('simulation_metadata', {
        'patient_name': patient_name_for_sim,
        'sampling_rate': initial_sampling_rate
    }, room=client_sid)

    while not stop_event.is_set():
        segment_count += 1
        app.logger.info(f"BG Task for {client_sid}: Starting segment {segment_count}.")
        sim_id_segment_base = f"sid_{client_sid}_seg_{segment_count}"

        # --- Run PPO Agent for the segment ---
        app.logger.info(f"BG Task for {client_sid}: Simulating PPO for segment {segment_count}")
        ppo_segment_data = run_simulation_segment(
            client_sid, PPO_MODEL_FILENAME_BASE, meal_data, "ppo",
            f"{sim_id_segment_base}_ppo", stop_event, patient_name_for_sim
        )
        if stop_event.is_set(): # Check immediately after potentially long call
            app.logger.info(f"BG Task for {client_sid}: Stop event detected after PPO segment. Ending loop.")
            break
        if ppo_segment_data and "error" in ppo_segment_data:
            app.logger.error(f"BG Task for {client_sid}: PPO Error in segment {segment_count}: {ppo_segment_data['error']}")
            # Error already emitted by run_simulation_segment
            break
        if ppo_segment_data is None: # Indicates interruption or unhandled error in run_simulation_segment
            app.logger.info(f"BG Task for {client_sid}: PPO segment {segment_count} returned None (interrupted or error). Ending loop.")
            break


        # --- Run PCPO Agent for the segment ---
        app.logger.info(f"BG Task for {client_sid}: Simulating PCPO for segment {segment_count}")
        pcpo_segment_data = run_simulation_segment(
            client_sid, PCPO_MODEL_FILENAME_BASE, meal_data, "pcpo",
            f"{sim_id_segment_base}_pcpo", stop_event, patient_name_for_sim
        )
        if stop_event.is_set(): # Check immediately
            app.logger.info(f"BG Task for {client_sid}: Stop event detected after PCPO segment. Ending loop.")
            break
        if pcpo_segment_data and "error" in pcpo_segment_data:
            app.logger.error(f"BG Task for {client_sid}: PCPO Error in segment {segment_count}: {pcpo_segment_data['error']}")
            break
        if pcpo_segment_data is None:
            app.logger.info(f"BG Task for {client_sid}: PCPO segment {segment_count} returned None (interrupted or error). Ending loop.")
            break

        # --- Interleave and Stream Data ---
        # More robust check for valid data before proceeding
        if not isinstance(ppo_segment_data, dict) or not isinstance(pcpo_segment_data, dict) or \
           'duration_steps' not in ppo_segment_data or 'duration_steps' not in pcpo_segment_data:
            err_msg = (f"Invalid segment data structure for segment {segment_count}. "
                       f"PPO data type: {type(ppo_segment_data)}, PCPO data type: {type(pcpo_segment_data)}")
            app.logger.error(f"BG Task for {client_sid}: {err_msg}")
            socketio.emit('simulation_error', {'error': err_msg}, room=client_sid)
            break

        num_steps_in_segment = ppo_segment_data.get('duration_steps', 0)
        if num_steps_in_segment == 0 or num_steps_in_segment != pcpo_segment_data.get('duration_steps', -1):
            err_msg = (f"Data length mismatch or zero length for segment {segment_count}. "
                       f"PPO steps: {num_steps_in_segment}, "
                       f"PCPO steps: {pcpo_segment_data.get('duration_steps', -1)}")
            app.logger.error(f"BG Task for {client_sid}: {err_msg}")
            socketio.emit('simulation_error', {'error': err_msg}, room=client_sid)
            break

        app.logger.info(f"BG Task for {client_sid}: Streaming combined data for segment {segment_count} ({num_steps_in_segment} steps).")
        for i in range(num_steps_in_segment):
            if stop_event.is_set():
                app.logger.info(f"BG Task for {client_sid}: Stop event detected during combined streaming of segment {segment_count}.")
                break

            try:
                combined_data_point = {
                    'step': global_step_offset + i,
                    'ppo_cgm': ppo_segment_data['cgm'][i],
                    'ppo_insulin': ppo_segment_data['insulin'][i],
                    'ppo_reward': ppo_segment_data['rewards'][i],
                    'pcpo_cgm': pcpo_segment_data['cgm'][i],
                    'pcpo_insulin': pcpo_segment_data['insulin'][i],
                    'pcpo_reward': pcpo_segment_data['rewards'][i],
                    'meal': ppo_segment_data['meal'][i], # Meal data is scenario-driven, should be same for both.
                }
                socketio.emit('simulation_data_point', combined_data_point, room=client_sid)
                socketio.sleep(0.005) # Slightly faster sleep as data is pre-computed for the segment
            except IndexError:
                app.logger.error(f"IndexError while creating combined_data_point at step {i} for segment {segment_count}. Data lengths might be inconsistent despite initial check.", exc_info=True)
                socketio.emit('simulation_error', {'error': 'Internal error: Inconsistent data during streaming.'}, room=client_sid)
                stop_event.set() # Stop the loop due to critical data error
                break
            except Exception as e_stream:
                app.logger.error(f"Error during data point emission: {e_stream}", exc_info=True)
                # Decide if this error is critical enough to stop everything
                # stop_event.set()
                break # Exit streaming loop for this segment


        if stop_event.is_set(): # Check again after the inner loop
            app.logger.info(f"BG Task for {client_sid}: Loop ending due to stop signal after streaming segment {segment_count}.")
            break

        global_step_offset += num_steps_in_segment
        socketio.emit('segment_complete', {'segment_number': segment_count, 'steps_in_segment': num_steps_in_segment}, room=client_sid)
        app.logger.info(f"BG Task for {client_sid}: Segment {segment_count} fully processed and streamed.")
        # socketio.sleep(0.1) # Optional small pause between full day segments

    app.logger.info(f"BG Task for {client_sid}: Continuous simulation loop finished.")
    socketio.emit('simulation_finished', {'message': 'Simulation stopped or completed.'}, room=client_sid)
    if client_sid in running_simulations_stop_events: # Cleanup from the dictionary
        del running_simulations_stop_events[client_sid]


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
        # The background task will clean up from the dictionary upon exiting its loop

@socketio.on('start_continuous_simulation')
def handle_start_continuous_simulation(data):
    client_sid = request.sid
    app.logger.info(f"Socket 'start_continuous_simulation' from {client_sid} with data: {data}")

    if client_sid in running_simulations_stop_events and \
       not running_simulations_stop_events[client_sid].is_set():
        socketio.emit('simulation_error', {"error": "A simulation is already running for your session. Please stop it first."}, room=client_sid)
        return

    try:
        if not data or 'meals' not in data:
            socketio.emit('simulation_error', {"error": "Missing 'meals' data in request."}, room=client_sid)
            return

        meal_details_input = data.get('meals', [])
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
                if time_min < 0 or carbs < 0:
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

        # Start the background task for the continuous simulation
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
        if client_sid in running_simulations_stop_events: # Ensure cleanup if error before task start
            running_simulations_stop_events[client_sid].set() # Signal stop just in case
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
            socketio.emit('simulation_stopping_ack', {'message': 'Stop signal received. Simulation will end after current step/segment.'}, room=client_sid)
        else:
            app.logger.info(f"Simulation for SID {client_sid} was already stopping or stopped.")
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

    if 'MAIN_PATH' not in os.environ:
         os.environ['MAIN_PATH'] = project_root_for_env
         logging.info(f"MAIN_PATH initially set from project root (fallback): {os.environ['MAIN_PATH']}")

    try:
        from decouple import config as decouple_config
        main_path_val_env = decouple_config('MAIN_PATH', default=os.environ['MAIN_PATH'])
        os.environ['MAIN_PATH'] = main_path_val_env
        logging.info(f"MAIN_PATH successfully set/confirmed: {main_path_val_env}")
    except Exception as e:
        logging.error(f"Error processing MAIN_PATH with decouple: {e}. MAIN_PATH remains: {os.environ.get('MAIN_PATH')}")
        if 'MAIN_PATH' not in os.environ:
            os.environ['MAIN_PATH'] = project_root_for_env
            logging.warning(f"MAIN_PATH was unexpectedly not set, re-setting to project root: {project_root_for_env}")


    app.config['VERBOSE_DEBUG'] = True

    socketio.run(app,
                 debug=True,
                 host='0.0.0.0',
                 port=int(os.environ.get("PORT", 5001))
                )