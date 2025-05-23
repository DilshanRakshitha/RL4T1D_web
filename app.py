import os
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from argparse import Namespace
from datetime import datetime, timedelta 
import logging # Standard Python logging
import shutil 

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

from agents.models.actor_critic import ActorCritic
from utils.worker import OnPolicyWorker

app = Flask(__name__) # Flask app instance
# Configure standard Python logging (used by Flask's app.logger too)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')


MODEL_DIR = "./models/" 
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}" 
PCPO_MODEL_FILENAME_BASE = "pcpo_model_for_patient_{}"
SIM_PATIENT_NAME_STR = '0' 
SIM_DURATION_MINUTES = 24 * 60
SIM_SAMPLING_RATE = 5

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logging.info(f"Created model directory: {MODEL_DIR}") # Use standard logging

def get_simulation_args(run_identifier_prefix="sim_run"):
    args = Namespace()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f") 
    unique_run_id = f"{run_identifier_prefix}_{timestamp}"

    args.experiment_folder = os.path.join("temp_web_sim_output", unique_run_id)
    args.experiment_dir = args.experiment_folder
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    # For app.config, it's better if this function is aware of the Flask app context,
    # or we pass verbose explicitly. For now, let's assume it can access app.
    # If run outside Flask context, this line would need adjustment.
    args.verbose = app.config.get('VERBOSE_DEBUG', True) if app else True # Check if app exists
    args.seed = random.randint(0, 100000)

    from environment.utils import get_patient_env 
    patients_from_util_with_hash, _ = get_patient_env()
    patients_from_util_no_hash = [p.replace("#", "") for p in patients_from_util_with_hash]

    try:
        args.patient_id = int(SIM_PATIENT_NAME_STR) # SIM_PATIENT_NAME_STR is global
        # Ensure the index is valid for the list we'll use for patient_name_for_env
        if not (0 <= args.patient_id < len(patients_from_util_with_hash)):
            raise ValueError(f"Patient ID {args.patient_id} derived from '{SIM_PATIENT_NAME_STR}' is out of bounds.")
        args.patient_name_for_env = patients_from_util_with_hash[args.patient_id]
        args.patient_name = str(args.patient_id) # The string '0', '1', etc. for consistency
        logging.info(f"Resolved SIM_PATIENT_NAME_STR '{SIM_PATIENT_NAME_STR}' to int_id {args.patient_id}, effective name for env: '{args.patient_name_for_env}'. String name for args.patient_name: '{args.patient_name}'")
    except ValueError as e:
        logging.error(f"CRITICAL: Error resolving SIM_PATIENT_NAME_STR '{SIM_PATIENT_NAME_STR}': {e}. Defaulting patient_id to 0.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "unknown_patient_fallback"
        args.patient_name = str(args.patient_id)


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
    args.meal_times_mean = [8.0, 10.5, 13.0, 16.5, 20.0, 22.5][:num_meal_slots]
    args.time_variance = [1e-8] * num_meal_slots; args.time_lower_bound = list(args.meal_times_mean)
    args.time_upper_bound = list(args.meal_times_mean); args.meal_prob = [-1.0] * num_meal_slots
    args.meal_amount = [0.0] * num_meal_slots; args.meal_variance = [1e-8] * num_meal_slots
    args.val_meal_prob = [-1.0] * num_meal_slots; args.val_meal_amount = [0.0] * num_meal_slots
    args.val_meal_variance = [1e-8] * num_meal_slots; args.val_time_variance = [1e-8] * num_meal_slots
    return args

def configure_custom_scenario_args(base_args, scenario_meal_data):
    worker_args = Namespace(**vars(base_args)) 
    # ... (rest of the function is fine, no app.logger calls) ...
    custom_meal_times_hours = []
    custom_meal_amounts_grams = []
    if scenario_meal_data:
        for meal_time_min, meal_carbs in scenario_meal_data:
            custom_meal_times_hours.append(float(meal_time_min) / 60.0)
            custom_meal_amounts_grams.append(float(meal_carbs))
    num_meal_slots = len(worker_args.meal_times_mean) 
    new_meal_times_mean = list(worker_args.meal_times_mean) 
    new_val_meal_amount = [0.0] * num_meal_slots 
    new_val_meal_prob = [-1.0] * num_meal_slots 
    for i in range(num_meal_slots):
        if i < len(custom_meal_times_hours):
            new_meal_times_mean[i] = custom_meal_times_hours[i]
            new_val_meal_amount[i] = custom_meal_amounts_grams[i]
            new_val_meal_prob[i] = 1.0 
    worker_args.meal_times_mean = new_meal_times_mean
    worker_args.val_meal_amount = new_val_meal_amount
    worker_args.val_meal_prob = new_val_meal_prob
    worker_args.val_time_variance = [1e-8] * num_meal_slots
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(worker_args.meal_times_mean) 
    worker_args.time_upper_bound = list(worker_args.meal_times_mean) 
    worker_args.env_type = 'testing' 
    return worker_args


def run_simulation_for_agent(model_filename_template, meal_data_for_sim, agent_type_suffix):
    actual_model_filename = model_filename_template.format(SIM_PATIENT_NAME_STR) # Uses global SIM_PATIENT_NAME_STR
    run_identifier = f"{SIM_PATIENT_NAME_STR}_{agent_type_suffix}"
    base_args = get_simulation_args(run_identifier_prefix=run_identifier)
    worker_args = configure_custom_scenario_args(base_args, meal_data_for_sim)
    
    actor_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Critic.pth")

    logging.info(f"Attempting to load Actor: {actor_path}") # Use standard logging
    logging.info(f"Attempting to load Critic: {critic_path}")

    if not os.path.exists(actor_path):
        logging.error(f"Actor model not found: {actor_path}")
        return {"error": f"Actor model not found: {actor_path}"}
    if not os.path.exists(critic_path):
        logging.error(f"Critic model not found: {critic_path}")
        return {"error": f"Critic model not found: {critic_path}"}

    policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
    policy_net.to(worker_args.device)
    policy_net.eval()

    api_worker_id = abs(hash(base_args.experiment_folder)) % 10000 + 8000 

    try:
        logging.info(f"Creating OnPolicyWorker with args for patient_name: '{worker_args.patient_name}' (int_id: {worker_args.patient_id})") # Use standard logging
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id)
        
        logging.info(f"Starting worker.rollout for patient_name '{worker_args.patient_name}', worker_id {api_worker_id}")
        worker.rollout(policy=policy_net, buffer=None) 
        logging.info(f"Finished worker.rollout for worker_id {api_worker_id}")
        
        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        
        logging.info(f"Attempting to read simulation log from: {log_file_path}")
        if not os.path.exists(log_file_path):
            logging.error(f"Simulation log file NOT FOUND: {log_file_path}")
            if os.path.exists(worker_args.worker_log_path_base):
                logging.error(f"Files in {worker_args.worker_log_path_base}: {os.listdir(worker_args.worker_log_path_base)}")
            else:
                logging.error(f"Log directory {worker_args.worker_log_path_base} does not exist.")
            return {"error": "Simulation log file not found after worker execution."}

        df = pd.read_csv(log_file_path) 
        if df.empty:
            logging.error(f"Simulation log file is empty: {log_file_path}")
            return {"error": "Simulation log file is empty."}

        column_map = {
            'cgm': 'cgm', 'insulin': 'ins', 'meals_input_per_step': 'meal', 'rewards': 'rew'
        }
        
        sim_results = {}
        for api_key, csv_col_name in column_map.items():
            if csv_col_name not in df.columns:
                logging.error(f"Expected column '{csv_col_name}' (for API key '{api_key}') not found in log file {log_file_path}. Available columns: {df.columns.tolist()}")
                return {"error": f"Missing column '{csv_col_name}' in simulation log."}
            sim_results[api_key] = df[csv_col_name].tolist()

        sim_results["total_reward"] = sum(sim_results["rewards"])
        sim_results["patient_name"] = worker_args.patient_name 
        sim_results["duration_steps"] = len(sim_results["cgm"])
    
    finally:
        if os.path.exists(base_args.experiment_folder):
            try:
                shutil.rmtree(base_args.experiment_folder)
                logging.info(f"Cleaned up temp directory: {base_args.experiment_folder}")
            except Exception as e:
                logging.error(f"Error cleaning up temp directory {base_args.experiment_folder}: {e}")
    
    return sim_results

def plot_simulation_results_to_base64(results_list, titles_list, main_title_patient_name):
    if not results_list or not any(r and not r.get("error") for r in results_list) or len(results_list) != len(titles_list):
        logging.warning("Plotting skipped: No valid results or mismatch in results/titles.") # Use standard logging
        return None
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    first_valid_result = next((r for r in results_list if r and not r.get("error")), None)
    if not first_valid_result: return None
    num_steps = len(first_valid_result["cgm"])
    time_in_hours = [t * SIM_SAMPLING_RATE / 60.0 for t in range(num_steps)]
    fig.suptitle(f"Simulation Comparison for Patient ID {main_title_patient_name}", fontsize=16)
    colors = ['blue', 'green', 'purple', 'orange']
    plotted_meals = False
    for i, (results_dict, title) in enumerate(zip(results_list, titles_list)):
        if not results_dict or results_dict.get("error"):
            logging.info(f"Skipping plot for {title} due to missing or error in results.") # Use standard logging
            continue
        color = colors[i % len(colors)]
        axs[0].plot(time_in_hours, results_dict["cgm"], label=f"{title} CGM", color=color, linewidth=1.5)
        axs[1].plot(time_in_hours, results_dict["insulin"], label=f"{title} Insulin", color=color, linestyle='--', linewidth=1.5)
        if not plotted_meals and results_dict.get("meals_input_per_step"):
            meal_values = results_dict["meals_input_per_step"]
            if meal_values : 
                meal_indices = [idx for idx, meal_val in enumerate(meal_values) if meal_val > 0]
                if meal_indices:
                    meal_cho_at_event = [meal_values[idx] for idx in meal_indices]
                    time_of_meals_in_hours = [time_in_hours[idx] for idx in meal_indices]
                    # MODIFICATION HERE: Removed use_line_collection=True
                    axs[2].stem(time_of_meals_in_hours, meal_cho_at_event, label="Meals CHO (g)", linefmt='grey', markerfmt='ko', basefmt=" ")
                    plotted_meals = True
    if not plotted_meals: axs[2].plot([], [], label="Meals CHO (g)") 
    axs[0].axhline(70, color='red', linestyle=':', linewidth=1.5, label='Hypo (70 mg/dL)')
    axs[0].axhline(180, color='orange', linestyle=':', linewidth=1.5, label='Hyper (180 mg/dL)')
    axs[0].set_ylabel("BG (mg/dL)", fontsize=12); axs[0].legend(loc='upper right', fontsize=10); axs[0].grid(True, linestyle=':', alpha=0.7); axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[1].set_ylabel("Insulin (U/step)", fontsize=12); axs[1].legend(loc='upper right', fontsize=10); axs[1].grid(True, linestyle=':', alpha=0.7); axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[2].set_ylabel("CHO (g)", fontsize=12); axs[2].set_xlabel(f"Time (hours) - Each step = {SIM_SAMPLING_RATE} min", fontsize=12); axs[2].legend(loc='upper right', fontsize=10); axs[2].grid(True, linestyle=':', alpha=0.7); axs[2].tick_params(axis='both', which='major', labelsize=10)
    if time_in_hours: axs[2].set_xlim(left=min(time_in_hours)-0.1 if time_in_hours else 0, right=max(time_in_hours)+0.1 if time_in_hours else 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    img = io.BytesIO(); plt.savefig(img, format='png', dpi=100); img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8'); plt.close(fig)
    return f"data:image/png;base64,{plot_url}"

@app.route('/simulate', methods=['POST'])
def simulate_glucose_endpoint():
    try:
        data = request.json
        if not data or 'meals' not in data: return jsonify({"error": "Missing 'meals' data in request."}), 400
        meal_details_input = data.get('meals', []) 
        parsed_meal_data = []
        for meal_item in meal_details_input:
            if "time_minutes" in meal_item and "carbs" in meal_item:
                parsed_meal_data.append((int(meal_item["time_minutes"]), float(meal_item["carbs"])))
            else: return jsonify({"error": "Each meal item must have 'time_minutes' and 'carbs'."}), 400
        parsed_meal_data.sort(key=lambda x: x[0])
        
        # Use app.logger within the Flask request context
        if app.config.get('VERBOSE_DEBUG', False): 
            app.logger.info(f"Received meal data for API: {parsed_meal_data}")

        ppo_model_file_template = PPO_MODEL_FILENAME_BASE 
        pcpo_model_file_template = PCPO_MODEL_FILENAME_BASE
        
        app.logger.info(f"Using PPO model template: {ppo_model_file_template}")
        app.logger.info(f"Using PCPO model template: {pcpo_model_file_template}")

        ppo_results = run_simulation_for_agent(ppo_model_file_template, parsed_meal_data, agent_type_suffix="ppo")
        pcpo_results = run_simulation_for_agent(pcpo_model_file_template, parsed_meal_data, agent_type_suffix="pcpo")
        
        plot_base64 = None
        patient_id_for_plot_title = SIM_PATIENT_NAME_STR 
        
        results_to_plot = []; titles_for_plot = []
        if ppo_results and not ppo_results.get("error"): results_to_plot.append(ppo_results); titles_for_plot.append("PPO")
        if pcpo_results and not pcpo_results.get("error"): results_to_plot.append(pcpo_results); titles_for_plot.append("PCPO")
        
        if results_to_plot: 
            plot_base64 = plot_simulation_results_to_base64(results_to_plot, titles_for_plot, patient_id_for_plot_title)
        
        return jsonify({"ppo_simulation": ppo_results, "pcpo_simulation": pcpo_results, "plot_image_base64": plot_base64})

    except FileNotFoundError as e: app.logger.error(f"A model file was not found: {e}", exc_info=True); return jsonify({"error": f"A required model file was not found: {str(e)}. Please check server configuration and model paths."}), 500
    except ValueError as e: app.logger.error(f"ValueError during simulation setup: {e}", exc_info=True); return jsonify({"error": f"Configuration error during simulation: {str(e)}"}), 500
    except Exception as e: app.logger.error(f"Error during simulation: {e}", exc_info=True); return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.config['VERBOSE_DEBUG'] = True 
    import decouple
    try:
        # Determine project root relative to this app.py file
        # If app.py is at the root of "rl4hcpo-rl4t1d", then this is correct.
        # If app.py is in a subdirectory of "rl4hcpo-rl4t1d", adjust accordingly.
        project_root_for_env = os.path.dirname(os.path.abspath(__file__)) 
        env_file = os.path.join(project_root_for_env, '.env')
        
        if not os.path.exists(env_file):
            with open(env_file, 'w') as f:
                f.write(f"MAIN_PATH={project_root_for_env}\n")
            logging.info(f"Created .env file at {env_file} with MAIN_PATH={project_root_for_env}") # Use standard logging
        else:
             logging.info(f".env file already exists at {env_file}. Ensure MAIN_PATH is correct.") # Use standard logging

        decouple.RepositoryEnv(env_file) 
        main_path_from_env = decouple.config('MAIN_PATH', default=project_root_for_env)
        logging.info(f"MAIN_PATH for decouple (from .env or default): {main_path_from_env}") # Use standard logging
        os.environ['MAIN_PATH'] = main_path_from_env
    except Exception as e:
        logging.warning(f"Could not setup/load .env for MAIN_PATH: {e}. Using CWD as fallback for MAIN_PATH if needed by decouple.") # Use standard logging
        if 'MAIN_PATH' not in os.environ: 
             os.environ['MAIN_PATH'] = os.getcwd()

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))