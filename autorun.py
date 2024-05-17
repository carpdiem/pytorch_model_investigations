# This file contains the necessary code for automatically detecting, profiling, running, and recording 
#   details about training runs from models for the MNIST Battleground

import utils
import os
import random
import psutil
import sys
import time
import importlib
import logging
import torch
logger = logging.getLogger(__name__)

def get_best_device():
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logging.info("MPS is available. Using MPS.")
    else:
        device = 'cpu'
        logging.info("CUDA and MPS are unavailable. Using CPU.")
    return device

def get_model_paths_from_dir(path, prefix='m_', suffix='.py'):
    file_list = os.listdir(path)

    filtered_files = [file for file in file_list if file.startswith(prefix) and file.endswith(suffix)]

    file_paths = [os.path.join(path, file) for file in filtered_files]

    return file_paths

def get_completed_runs(path, completed_flag='completed.txt', prefix='m_'):
    dir_list = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

    filtered_dirs = [dir for dir in dir_list if dir.startswith(prefix)]

    completed_dirs = [dir for dir in filtered_dirs if os.path.isfile(os.path.join(path, dir, completed_flag))]

    return completed_dirs

def get_model_paths_to_train(all_model_paths, completed_models, randomized_order = True):
    remaining_model_paths = [
        model_path for model_path in all_model_paths
        if os.path.splitext(os.path.basename(model_path))[0] not in completed_models
    ]

    if randomized_order:
        random.shuffle(remaining_model_paths)

    return remaining_model_paths

def already_running():
    current_script_path = os.path.abspath(sys.argv[0])
    
    for process in psutil.process_iter(['pid', 'name']):
        try:
            if process.pid != os.getpid():
                process_cmdline = process.cmdline()
                ## The line below this IS NOT A TYPO. we check for 'ython' to ensure
                ##   compatibility on both macos (which invokes 'Python') and linux (which invokes 'python3')
                if any('ython' in arg for arg in process_cmdline):
                    process_script_path = process_cmdline[1]
                    if os.path.abspath(process_script_path) == current_script_path:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, IndexError):
            pass
    
    return False

def load_model_and_train(model_path, time_allotted_seconds = 4 * 60 * 60):
    model_dir, model_file = os.path.split(model_path)
    model_name = os.path.splitext(model_file)[0]
    os.sys.path.append(model_dir)

    device = get_best_device()

    try:
        logging.info(f"Importing model {model_name}")
        model_module = importlib.import_module(model_name)

        if hasattr(model_module, 'DeepNet'):
            model = model_module.DeepNet().to(device)
            logging.info(f"Model {model_name} initialized successfully.")
            
            # determine time for one epoch
            start_time = time.time()
            utils.loss_vs_flops(model, epochs=1, device=device)
            epoch_time = time.time() - start_time
            logging.info(f"Time for one epoch: ~{epoch_time} seconds")

            # determine total number of training epochs to be performed
            num_epochs = int(time_allotted_seconds / epoch_time)
            predicted_duration = num_epochs * epoch_time
            logging.info(f"Training for {num_epochs} epochs")
            logging.info(f"Expected training time: ~{predicted_duration / 60 / 60} hours, aka {predicted_duration} seconds")


            # train the model
            logging.info(f"Beginning Training for {model_name}")
            start_time = time.time()
            model = model_module.DeepNet().to(device)
            res = utils.loss_vs_flops(model, epochs=num_epochs, device=device)
            logging.info(f"Training completed for {model_name} in {time.time() - start_time} seconds")

            # create the results directory
            logging.info(f"Creating results directory for {model_name}")
            dirpath = os.path.join('models', 'results', model_name)
            os.makedirs(dirpath, exist_ok=True)

            # save the trained model
            logging.info(f"Saving trained model as {model_name}.pth")
            torch.save(res.pop('model').state_dict(), os.path.join(dirpath, model_name) + '.pth')

            # record the results as a csv
            logging.info(f"Saving results as {model_name}.csv")
            utils.write_results_to_csv(res, os.path.join(dirpath, model_name) + '.csv')

            # write a completed.txt file to the results directory
            with open(os.path.join(dirpath, 'completed.txt'), 'w') as f:
                logging.info(f"Writing completion file for {model_name}")
                f.write(f"Completed training for {model_name}\n")
                f.write(f"Predicted duration was: {predicted_duration} seconds\n")
                f.write(f"Actual Duration was: {time.time() - start_time} seconds\n")
                f.write(f"Number of Epochs run: {num_epochs}\n")
                f.write(f"Final accuracy: {res['accuracies'][-1]}\n")

            logging.info(f"All training activities complete for {model_name}")
        else:
            logging.error(f"DeepNet class not found in {model_name}.")

    except ImportError:
        print(f"Failed to import model {model_name}.")

    os.sys.path.remove(model_dir)

def main(time_allotted_seconds = 4 * 60 * 60):
    logging.basicConfig(filename='autorun.log',
                        filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')
    logging.info("Starting autorun.py")
    if not already_running():
        logging.info("Not running yet!")
        model_paths = get_model_paths_from_dir('models')
        completed_runs = get_completed_runs('models/results')
        model_paths_to_train = get_model_paths_to_train(model_paths, completed_runs)
        logging.info(f"Models to train: {model_paths_to_train}")
        for model_path in model_paths_to_train:
            logging.info(f"Training model {model_path}")
            load_model_and_train(model_path, time_allotted_seconds=time_allotted_seconds)
        logging.info("All models trained!")
        logging.info("exiting now")
    else:
        logging.info("Script already running!")
        logging.info("exiting now")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(time_allotted_seconds=int(sys.argv[1]))
    else:
        main()
