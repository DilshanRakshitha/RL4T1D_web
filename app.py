import os
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template
from argparse import Namespace
from datetime import datetime
import logging
import shutil
import uuid # For unique simulation IDs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io

from agents.models.actor_critic import ActorCritic
from utils.worker import OnPolicyWorker

app = Flask(__name__, template_folder='templates')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(filename)s:%(lineno)d %(message)s')

MODEL_DIR = "./models/"
PPO_MODEL_FILENAME_BASE = "ppo_model_for_patient_{}"
PCPO_MODEL_FILENAME_BASE = "pcpo_model_for_patient_{}"
SIM_PATIENT_NAME_STR = '0'
SIM_DURATION_MINUTES = 24 * 60
SIM_SAMPLING_RATE = 5

# --- Global Cache for Simulation Data ---
simulation_cache = {}
# ---

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- (get_simulation_args, configure_custom_scenario_args, run_simulation_for_agent, plot_simulation_results_to_file
#      are largely the same as in the previous correct version, with minor adjustments for unique run IDs if needed) ---

def get_simulation_args(run_identifier_prefix="sim_run"):
    args = Namespace()
    # Use a consistent base for experiment_folder that might be shared if not careful
    # but worker_log_path_base should be unique per full simulation
    unique_log_id = f"{run_identifier_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    args.experiment_folder = os.path.join("temp_web_sim_output", unique_log_id) # Made unique
    args.experiment_dir = args.experiment_folder
    args.worker_log_path_base = os.path.join(args.experiment_dir, "testing", "data")
    os.makedirs(args.worker_log_path_base, exist_ok=True, mode=0o755)

    args.device = "cpu"
    args.verbose = app.config.get('VERBOSE_DEBUG', False) # Less verbose for this
    args.seed = random.randint(0, 100000)
    from environment.utils import get_patient_env
    patients_from_util_with_hash, _ = get_patient_env()
    try:
        args.patient_id = int(SIM_PATIENT_NAME_STR)
        if not (0 <= args.patient_id < len(patients_from_util_with_hash)):
            raise ValueError(f"Patient ID {args.patient_id} out of bounds.")
        args.patient_name_for_env = patients_from_util_with_hash[args.patient_id]
        args.patient_name = str(args.patient_id)
    except ValueError as e:
        logging.error(f"Error resolving SIM_PATIENT_NAME_STR: {e}. Defaulting.")
        args.patient_id = 0
        args.patient_name_for_env = patients_from_util_with_hash[0] if patients_from_util_with_hash else "patient#000"
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
    num_meal_slots = 6 # Max 6 meals
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
    
    num_meal_slots = len(worker_args.meal_times_mean) # Use the length from default args
    
    # Initialize with defaults, then overwrite
    new_meal_times_mean = list(worker_args.meal_times_mean)
    new_val_meal_amount = [0.0] * num_meal_slots # Default to 0 amount
    new_val_meal_prob = [-1.0] * num_meal_slots   # Default to no meal (prob -1)

    for i in range(len(custom_meal_times_hours)):
        if i < num_meal_slots: # Ensure we don't go out of bounds for the scenario config
            new_meal_times_mean[i] = custom_meal_times_hours[i]
            new_val_meal_amount[i] = custom_meal_amounts_grams[i]
            new_val_meal_prob[i] = 1.0 # Meal occurs
        else:
            # This case implies more meals provided than slots in base config,
            # which shouldn't happen if num_meal_slots is fixed (e.g., 6).
            # If it can, then the base config needs to be large enough or dynamic.
            logging.warning(f"Meal entry {i+1} exceeds available scenario meal slots ({num_meal_slots}). Ignoring.")


    worker_args.meal_times_mean = new_meal_times_mean
    worker_args.val_meal_amount = new_val_meal_amount
    worker_args.val_meal_prob = new_val_meal_prob
    
    # Ensure these are lists of the correct length (num_meal_slots)
    worker_args.val_time_variance = [1e-8] * num_meal_slots
    worker_args.val_meal_variance = [1e-8] * num_meal_slots
    worker_args.time_lower_bound = list(new_meal_times_mean) # Match mean for fixed times
    worker_args.time_upper_bound = list(new_meal_times_mean) # Match mean for fixed times
    
    worker_args.env_type = 'testing'
    return worker_args

def run_simulation_for_agent(model_filename_template, meal_data_for_sim, agent_type_suffix, sim_id_for_logging):
    actual_model_filename = model_filename_template.format(SIM_PATIENT_NAME_STR)
    # Use sim_id_for_logging to make folder unique for this full simulation run's logs
    base_args = get_simulation_args(run_identifier_prefix=f"{sim_id_for_logging}_{agent_type_suffix}")
    worker_args = configure_custom_scenario_args(base_args, meal_data_for_sim)

    actor_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Actor.pth")
    critic_path = os.path.join(MODEL_DIR, f"{actual_model_filename}_Critic.pth")

    if not os.path.exists(actor_path): return {"error": f"Actor model not found: {actor_path}"}
    if not os.path.exists(critic_path): return {"error": f"Critic model not found: {critic_path}"}

    policy_net = ActorCritic(args=worker_args, load=True, actor_path=actor_path, critic_path=critic_path)
    policy_net.to(worker_args.device); policy_net.eval()

    api_worker_id = abs(hash(sim_id_for_logging + agent_type_suffix)) % 10000 + 8000 # Consistent worker ID for this sim_id part
    
    sim_results = {"error": "Simulation did not complete fully or log file issue."} # Default error
    try:
        logging.info(f"Running {agent_type_suffix} for sim ID {sim_id_for_logging}, worker_id {api_worker_id}, log path base: {worker_args.worker_log_path_base}")
        worker = OnPolicyWorker(args=worker_args, env_args=worker_args, mode='testing', worker_id=api_worker_id)
        worker.rollout(policy=policy_net, buffer=None)
        
        log_file_path = os.path.join(worker_args.worker_log_path_base, f"logs_worker_{api_worker_id}.csv")
        logging.info(f"Log file should be at: {log_file_path}")

        if not os.path.exists(log_file_path):
            logging.error(f"Simulation log file NOT FOUND: {log_file_path}")
            sim_results = {"error": f"Sim log not found: {log_file_path}"}
        else:
            df = pd.read_csv(log_file_path)
            if df.empty:
                logging.error(f"Simulation log file is empty: {log_file_path}")
                sim_results = {"error": "Simulation log is empty."}
            else:
                if 'meals_input_per_step' not in df.columns:
                    logging.warning(f"'meals_input_per_step' not in {log_file_path}, defaulting to 0.")
                    df['meals_input_per_step'] = 0.0

                # Check for essential columns
                essential_cols = {'cgm': 'cgm', 'insulin': 'ins', 'rewards': 'rew'}
                for api_key, csv_col_name in essential_cols.items():
                    if csv_col_name not in df.columns:
                        logging.error(f"Missing essential column '{csv_col_name}' in {log_file_path}")
                        return {"error": f"Missing column '{csv_col_name}' in simulation log."}

                sim_results = {
                    'cgm': df['cgm'].tolist(),
                    'insulin': df['ins'].tolist(),
                    'meal': df['meals_input_per_step'].tolist(), # Use the (potentially defaulted) column
                    'rewards': df['rew'].tolist()
                }
                sim_results["total_reward"] = sum(sim_results.get("rewards", []))
                sim_results["patient_name"] = worker_args.patient_name
                sim_results["duration_steps"] = len(sim_results.get("cgm", []))
    except Exception as e:
        logging.error(f"Exception during {agent_type_suffix} simulation for {sim_id_for_logging}: {e}", exc_info=True)
        sim_results = {"error": f"Exception during simulation: {str(e)}"}
    finally:
        # Cleanup the unique folder for this simulation run
        folder_to_cleanup = getattr(base_args, 'experiment_folder', None) # experiment_folder is unique due to unique_log_id
        if folder_to_cleanup and os.path.exists(folder_to_cleanup):
            try:
                shutil.rmtree(folder_to_cleanup)
                logging.info(f"Cleaned up temp directory: {folder_to_cleanup}")
            except Exception as e:
                logging.error(f"Error cleaning up temp directory {folder_to_cleanup}: {e}")
    return sim_results

def plot_simulation_results_to_file(results_list, titles_list, main_title_patient_id, save_path_prefix):
    # (This function remains the same as your last provided correct version for server-side saving)
    # It's good practice to ensure it handles cases where results_list might contain error dicts.
    valid_results_for_plot = []
    valid_titles_for_plot = []
    for res, title in zip(results_list, titles_list):
        if res and not res.get("error") and res.get("cgm"):
            valid_results_for_plot.append(res)
            valid_titles_for_plot.append(title)

    if not valid_results_for_plot:
        logging.warning("Plotting to file skipped: No valid results with CGM data.")
        return

    # ... (rest of your existing plotting logic using valid_results_for_plot and valid_titles_for_plot) ...
    fig = plt.figure(figsize=(16, 8))
    ax2 = fig.add_subplot(111)
    divider = make_axes_locatable(ax2)
    ax1 = divider.append_axes("top", size="250%", pad=0.0, sharex=ax2)
    fig.subplots_adjust(hspace=0.05)
    ax1.set_ylabel('CGM [mg/dL]', color='#000080', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#000080', labelsize=10)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.axhspan(70, 180, alpha=0.2, color='limegreen', lw=0, label='Normoglycemia (70-180)')
    ax1.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='Hypoglycemia (<70)')
    ax1.axhline(y=180, color='darkorange', linestyle='--', linewidth=1.5, label='Hyperglycemia (>180)')
    ax1.axhline(y=250, color='maroon', linestyle=':', linewidth=1.5, label='Severe Hyper (>250)')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_ylabel('Insulin [U/step]', color='mediumseagreen', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='mediumseagreen', labelsize=10)
    ax2.set_xlabel(f"Time (hours) - Each step = {SIM_SAMPLING_RATE} min", fontsize=12)
    ax2.tick_params(axis='x', labelsize=10)
    ax3_meals = ax2.twinx()
    ax3_meals.set_ylabel('CHO (g)', color='#800000', fontsize=12)
    ax3_meals.tick_params(axis='y', labelcolor='#800000', labelsize=10)

    first_valid_result = valid_results_for_plot[0] # Already checked it exists
    num_steps = len(first_valid_result["cgm"])
    time_in_hours = [t * SIM_SAMPLING_RATE / 60.0 for t in range(num_steps)]
    fig.suptitle(f"Simulation Results for Patient ID {main_title_patient_id}", fontsize=16, y=0.99)
    plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plotted_meals_overall = False
    max_cgm_val_overall = 0; max_ins_val_overall = 0; max_cho_val_overall = 0

    for i, (results_dict, title) in enumerate(zip(valid_results_for_plot, valid_titles_for_plot)):
        # Already know results_dict is valid and has "cgm"
        color = plot_colors[i % len(plot_colors)]
        ax1.plot(time_in_hours, results_dict["cgm"], label=f"{title} CGM", color=color, linewidth=1.5)
        if results_dict["cgm"]: max_cgm_val_overall = max(max_cgm_val_overall, max(results_dict["cgm"]))
        
        current_insulin_data = results_dict.get("insulin", [])
        if current_insulin_data:
            ax2.plot(time_in_hours, current_insulin_data, label=f"{title} Insulin", color=color, linestyle='--', linewidth=1.5)
            max_ins_val_overall = max(max_ins_val_overall, max(current_insulin_data) if current_insulin_data else 0)

        if not plotted_meals_overall and results_dict.get("meal"):
            meal_values = results_dict["meal"]
            if meal_values:
                if any(m > 0 for m in meal_values): max_cho_val_overall = max(max_cho_val_overall, max(m for m in meal_values if m > 0))
                meal_indices = [idx for idx, meal_val in enumerate(meal_values) if meal_val > 0]
                if meal_indices:
                    meal_cho_at_event = [meal_values[idx] for idx in meal_indices]
                    time_of_meals_in_hours = [time_in_hours[idx] for idx in meal_indices]
                    
                    # Annotations
                    meal_annotation_toggle = True
                    current_max_cgm_for_annotation = max(max_cgm_val_overall, 300)
                    cgm_for_annotation = first_valid_result.get("cgm", []) # Use CGM from first valid for positioning
                    if cgm_for_annotation:
                        for meal_idx_in_list, cho in enumerate(meal_cho_at_event):
                            original_meal_idx = meal_indices[meal_idx_in_list]
                            time_h = time_in_hours[original_meal_idx]
                            cgm_at_meal_for_annotation_base = cgm_for_annotation[original_meal_idx]
                            offset_y_val = (current_max_cgm_for_annotation * 0.90) if meal_annotation_toggle else (current_max_cgm_for_annotation * 0.80)
                            offset_y_val = max(cgm_at_meal_for_annotation_base + 30, min(offset_y_val, current_max_cgm_for_annotation - 10))
                            ax1.annotate(f'{cho:.0f}g', (time_h, offset_y_val), xytext=(0,5), textcoords='offset points', color='#800000', ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3, ec="none"))
                            ax1.plot([time_h, time_h], [cgm_at_meal_for_annotation_base, offset_y_val-5], color='#800000', linestyle='-', linewidth=0.8, alpha=0.7)
                            meal_annotation_toggle = not meal_annotation_toggle

                    markerline, stemlines, baseline = ax3_meals.stem(time_of_meals_in_hours, meal_cho_at_event, linefmt='grey', markerfmt='o', basefmt=" ")
                    plt.setp(markerline, markerfacecolor='#800000', markeredgecolor='grey', markersize=5)
                    markerline.set_label("Meals CHO (g)")
                    plotted_meals_overall = True
    
    if not plotted_meals_overall:
        dummy_meal_line = mlines.Line2D([], [], color='#800000', marker='o', linestyle='None', markersize=5, label='Meals CHO (g)')
        ax3_meals.add_line(dummy_meal_line)

    ax1.set_ylim(0, max(max_cgm_val_overall * 1.1, 350))
    ax2.set_ylim(0, max(max_ins_val_overall * 1.2, 1.0 if max_ins_val_overall == 0 else max_ins_val_overall * 1.2))
    ax3_meals.set_ylim(0, max(max_cho_val_overall * 1.1, 10.0 if max_cho_val_overall == 0 else max_cho_val_overall * 1.1))

    handles, labels = [], []
    for ax_loop in [ax1, ax2, ax3_meals]:
        for handle, label in zip(*ax_loop.get_legend_handles_labels()):
            if label not in labels: handles.append(handle); labels.append(label)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=max(1,len(handles)//2 + len(handles)%2), fontsize=9, frameon=False)
    if time_in_hours: ax2.set_xlim(left=min(time_in_hours)-0.1 if time_in_hours else 0, right=max(time_in_hours)+0.1 if time_in_hours else 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    if save_path_prefix:
        img_bytes_for_saving = io.BytesIO()
        plt.savefig(img_bytes_for_saving, format='png', dpi=120)
        img_bytes_for_saving.seek(0)
        save_dir = os.path.dirname(save_path_prefix)
        if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path_prefix}_patientID_{main_title_patient_id}_{timestamp_str}.png"
        try:
            with open(filename, "wb") as f_out: f_out.write(img_bytes_for_saving.getvalue())
            logging.info(f"Plot saved to {filename}")
        except Exception as e: logging.error(f"Failed to save plot to {filename}: {e}")
    plt.close(fig)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation_endpoint():
    try:
        data = request.json
        if not data or 'meals' not in data:
            return jsonify({"error": "Missing 'meals' data."}), 400
        
        meal_details_input = data.get('meals', [])
        parsed_meal_data = []
        for meal_item in meal_details_input:
            # Basic validation for meal_item structure
            if not isinstance(meal_item, dict) or "time_minutes" not in meal_item or "carbs" not in meal_item:
                return jsonify({"error": "Meal items must be objects with 'time_minutes' and 'carbs'."}), 400
            try:
                time_min = int(meal_item["time_minutes"])
                carbs = float(meal_item["carbs"])
                if time_min < 0 or carbs < 0:
                     return jsonify({"error": "Meal time and carbs cannot be negative."}), 400
                parsed_meal_data.append((time_min, carbs))
            except ValueError:
                return jsonify({"error": "Invalid number format for meal time or carbs."}), 400
        parsed_meal_data.sort(key=lambda x: x[0])

        sim_id = str(uuid.uuid4())
        app.logger.info(f"Starting simulation with ID: {sim_id} for meals: {parsed_meal_data}")

        # Run simulations and store in cache
        ppo_results = run_simulation_for_agent(PPO_MODEL_FILENAME_BASE, parsed_meal_data, "ppo", sim_id)
        pcpo_results = run_simulation_for_agent(PCPO_MODEL_FILENAME_BASE, parsed_meal_data, "pcpo", sim_id)
        
        simulation_cache[sim_id] = {"ppo_simulation": ppo_results, "pcpo_simulation": pcpo_results}
        
        total_steps = 0
        patient_name_from_sim = SIM_PATIENT_NAME_STR # Default
        warning_message = None

        # Determine total_steps and patient_name from successful simulations
        if ppo_results and "error" not in ppo_results and ppo_results.get("duration_steps"):
            total_steps = ppo_results["duration_steps"]
            patient_name_from_sim = ppo_results.get("patient_name", patient_name_from_sim)
        elif pcpo_results and "error" not in pcpo_results and pcpo_results.get("duration_steps"):
            total_steps = pcpo_results["duration_steps"]
            patient_name_from_sim = pcpo_results.get("patient_name", patient_name_from_sim)
        else: # Fallback if both fail or lack duration_steps
            total_steps = SIM_DURATION_MINUTES // SIM_SAMPLING_RATE
            logging.warning(f"Could not determine total_steps from simulation for {sim_id}, using default.")
        
        if (ppo_results and "error" in ppo_results) or \
           (pcpo_results and "error" in pcpo_results):
            warning_message = "One or both simulations encountered an error. Results may be incomplete."
            if ppo_results and "error" in ppo_results: app.logger.error(f"PPO Sim Error (ID: {sim_id}): {ppo_results['error']}")
            if pcpo_results and "error" in pcpo_results: app.logger.error(f"PCPO Sim Error (ID: {sim_id}): {pcpo_results['error']}")


        # Save combined plot to file (optional, for server-side records)
        results_to_plot_for_saving = []
        titles_for_plot_saving = []
        if ppo_results and "error" not in ppo_results:
            results_to_plot_for_saving.append(ppo_results)
            titles_for_plot_saving.append("PPO")
        if pcpo_results and "error" not in pcpo_results:
            results_to_plot_for_saving.append(pcpo_results)
            titles_for_plot_saving.append("PCPO")
        
        if results_to_plot_for_saving:
            plot_save_directory = "plots" # Ensure this directory exists or is created
            if not os.path.exists(plot_save_directory):
                os.makedirs(plot_save_directory)
            save_prefix = os.path.join(plot_save_directory, f"sim_plot_combined_{sim_id}")
            plot_simulation_results_to_file(results_to_plot_for_saving, titles_for_plot_saving, patient_name_from_sim, save_prefix)


        response_payload = {
            "simulation_id": sim_id,
            "total_steps": total_steps,
            "sim_sampling_rate": SIM_SAMPLING_RATE,
            "patient_name": patient_name_from_sim
        }
        if warning_message:
            response_payload["warning"] = warning_message
            
        return jsonify(response_payload)

    except Exception as e:
        app.logger.error(f"Error in /start_simulation: {e}", exc_info=True)
        # Ensure sim_id is not in cache if we failed before putting anything in
        sim_id_ref = locals().get('sim_id', None)
        if sim_id_ref and sim_id_ref in simulation_cache:
            del simulation_cache[sim_id_ref]
        return jsonify({"error": f"Unexpected server error during simulation setup: {str(e)}"}), 500


@app.route('/get_simulation_chunk', methods=['GET'])
def get_simulation_chunk_endpoint():
    sim_id = request.args.get('simulation_id')
    start_index = int(request.args.get('start_index', 0))
    chunk_size = int(request.args.get('chunk_size', 50))

    if not sim_id or sim_id not in simulation_cache:
        return jsonify({"error": "Invalid or expired simulation ID."}), 404

    full_sim_data = simulation_cache[sim_id]
    ppo_full = full_sim_data.get("ppo_simulation", {})
    pcpo_full = full_sim_data.get("pcpo_simulation", {})
    
    # Helper to create a chunk for an agent
    def create_chunk(agent_full_data):
        chunk = {"cgm": [], "insulin": [], "meal": []} # Ensure keys exist
        if agent_full_data and "error" not in agent_full_data and agent_full_data.get("cgm"):
            chunk["cgm"] = agent_full_data["cgm"][start_index : start_index + chunk_size]
            chunk["insulin"] = agent_full_data.get("insulin", [])[start_index : start_index + chunk_size]
            chunk["meal"] = agent_full_data.get("meal", [])[start_index : start_index + chunk_size]
        elif agent_full_data and "error" in agent_full_data:
            chunk["error_message"] = agent_full_data["error"]
        return chunk

    ppo_chunk = create_chunk(ppo_full)
    pcpo_chunk = create_chunk(pcpo_full)

    # Determine total_steps based on available valid data
    total_steps_ppo = len(ppo_full.get("cgm", [])) if ppo_full and "error" not in ppo_full else 0
    total_steps_pcpo = len(pcpo_full.get("cgm", [])) if pcpo_full and "error" not in pcpo_full else 0
    effective_total_steps = max(total_steps_ppo, total_steps_pcpo, SIM_DURATION_MINUTES // SIM_SAMPLING_RATE) # Fallback

    is_final_chunk = (start_index + chunk_size) >= effective_total_steps
    
    # Optional: Clean up cache earlier if no longer needed by any client.
    # For simplicity, we might let them expire or clean up manually if memory is an issue.
    # if is_final_chunk and sim_id in simulation_cache:
    #     del simulation_cache[sim_id] 
    #     app.logger.info(f"Removed simulation {sim_id} from cache after final chunk.")

    return jsonify({
        "ppo_chunk": ppo_chunk,
        "pcpo_chunk": pcpo_chunk,
        "is_final_chunk": is_final_chunk,
        "next_start_index": start_index + chunk_size,
        "actual_total_steps": effective_total_steps # Send this to help client know true end
    })


if __name__ == '__main__':
    app.config['VERBOSE_DEBUG'] = True
    import decouple
    try:
        project_root_for_env = os.path.dirname(os.path.abspath(__file__))
        env_file = os.path.join(project_root_for_env, '.env')
        if not os.path.exists(env_file):
            with open(env_file, 'w') as f: f.write(f"MAIN_PATH={project_root_for_env}\n")
        decouple.RepositoryEnv(env_file)
        main_path_from_env = decouple.config('MAIN_PATH', default=project_root_for_env)
        os.environ['MAIN_PATH'] = main_path_from_env
    except Exception as e:
        logging.warning(f"Decouple .env setup failed: {e}. MAIN_PATH might not be set from .env.")
        if 'MAIN_PATH' not in os.environ: os.environ['MAIN_PATH'] = os.getcwd()

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))