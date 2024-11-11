import os
import re
import pandas as pd

param_pattern = re.compile(
r"tensor\(\[([\d.]+), ([\d.]+)\], device='cuda:0', requires_grad=True\)")

def parse_output_hiv(file):
    # Define regex patterns for train, validation, and test metrics, along with parameter values
    train_pattern = re.compile(
        r"train: \{.*?'epoch': (\d+).*?'time_epoch': ([\d.]+),.*?'eta': ([\d.]+),.*?'eta_hours': ([\d.]+),"
        r".*?'loss': ([\d.]+),.*?'lr': ([\de\-.]+),.*?'params': (\d+),.*?'time_iter': ([\d.]+),"
        r".*?'accuracy': ([\d.]+),.*?'precision': ([\d.]+),.*?'recall': ([\d.]+),"
        r".*?'f1': ([\d.]+),.*?'auc': ([\d.]+)"
    )
    val_pattern = re.compile(
        r"val: \{.*?'epoch': (\d+).*?'time_epoch': ([\d.]+),.*?'loss': ([\d.]+),.*?'lr': ([\d.]+),"
        r".*?'params': (\d+),.*?'time_iter': ([\d.]+),.*?'accuracy': ([\d.]+),"
        r".*?'precision': ([\d.]+),.*?'recall': ([\d.]+),.*?'f1': ([\d.]+),"
        r".*?'auc': ([\d.]+)"
    )
    test_pattern = re.compile(
        r"test: \{.*?'epoch': (\d+).*?'time_epoch': ([\d.]+),.*?'loss': ([\d.]+),.*?'lr': ([\d.]+),"
        r".*?'params': (\d+),.*?'time_iter': ([\d.]+),.*?'accuracy': ([\d.]+),"
        r".*?'precision': ([\d.]+),.*?'recall': ([\d.]+),.*?'f1': ([\d.]+),"
        r".*?'auc': ([\d.]+)"
    )

    # Initialize data storage
    data = []
    epoch_data = {'epoch': None}
    param_pairs = []  # Track pairs of MPNN and Global Attention parameters for each layer
    train_found = False
    val_found = False
    test_found = False
    with open(file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Extract training metrics
            train_match = train_pattern.search(line)
            if train_match:
                train_found = True
                epoch_data.update({
                    'epoch': int(train_match.group(1)),
                    'train_time_epoch': float(train_match.group(2)),
                    'train_eta': float(train_match.group(3)),
                    'train_eta_hours': float(train_match.group(4)),
                    'train_loss': float(train_match.group(5)),
                    'train_lr': float(train_match.group(6)),
                    'train_params': int(train_match.group(7)),
                    'train_time_iter': float(train_match.group(8)),
                    'train_accuracy': float(train_match.group(9)),
                    'train_precision': float(train_match.group(10)),
                    'train_recall': float(train_match.group(11)),
                    'train_f1': float(train_match.group(12)),
                    'train_auc': float(train_match.group(13))
                })

            # Extract validation metrics
            val_match = val_pattern.search(line)
            if val_match:
                val_found = True
                epoch_data.update({
                    'epoch': int(val_match.group(1)),
                    'val_time_epoch': float(val_match.group(2)),
                    'val_loss': float(val_match.group(3)),
                    'val_lr': float(val_match.group(4)),
                    'val_params': int(val_match.group(5)),
                    'val_time_iter': float(val_match.group(6)),
                    'val_accuracy': float(val_match.group(7)),
                    'val_precision': float(val_match.group(8)),
                    'val_recall': float(val_match.group(9)),
                    'val_f1': float(val_match.group(10)),
                    'val_auc': float(val_match.group(11))
                })

            # Extract test metrics
            test_match = test_pattern.search(line)
            if test_match:
                test_found = True
                epoch_data.update({
                    'epoch': int(test_match.group(1)),
                    'test_time_epoch': float(test_match.group(2)),
                    'test_loss': float(test_match.group(3)),
                    'test_lr': float(test_match.group(4)),
                    'test_params': int(test_match.group(5)),
                    'test_time_iter': float(test_match.group(6)),
                    'test_accuracy': float(test_match.group(7)),
                    'test_precision': float(test_match.group(8)),
                    'test_recall': float(test_match.group(9)),
                    'test_f1': float(test_match.group(10)),
                    'test_auc': float(test_match.group(11))
                })

            # Extract parameters
            param_match = param_pattern.search(line)
            if param_match:
                mpnn_value = float(param_match.group(1))
                global_attention_value = float(param_match.group(2))
                param_pairs.append({'mpnn_param': mpnn_value, 'global_attention_param': global_attention_value})
                
            # Store epoch data when all metrics and parameters are captured
            if val_found and train_found and test_found:
                if (len(param_pairs) == 10 and filename in ['molhiv_GPS.out', 'molhiv_GPS+RWSE.out']) or (len(param_pairs) == 2 and filename == 'molhiv_GPS+RWSEdev.out'):
                # Flatten param pairs into columns for the DataFrame
                    for i, params in enumerate(param_pairs):
                        epoch_data[f'mpnn_param_layer_{i+1}'] = params['mpnn_param']
                        epoch_data[f'global_attention_param_layer_{i+1}'] = params['global_attention_param']
                    data.append(epoch_data.copy())
                    epoch_data = {'epoch': None}
                    param_pairs = []
                    train_found = False
                    val_found = False
                    test_found = False
        # Convert list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
    return df

# Function to extract data and store it into a single dataframe
def parse_output_pcba(filename):
    # Define regular expression patterns
    train_pattern = re.compile(
        r"train: \{'epoch': (\d+), 'time_epoch': ([\d.]+), 'eta': ([\d.]+), 'eta_hours': ([\d.]+), 'loss': ([\d.]+), 'lr': ([\d.]+), 'params': (\d+), 'time_iter': ([\d.]+), 'accuracy': ([\d.]+), 'auc': ([\d.]+), 'ap': ([\d.]+)\}"
    )

    val_pattern = re.compile(
        r"val: \{'epoch': (\d+), 'time_epoch': ([\d.]+), 'loss': ([\d.]+), 'lr': ([\d.]+), 'params': (\d+), 'time_iter': ([\d.]+), 'accuracy': ([\d.]+), 'auc': ([\d.]+), 'ap': ([\d.]+)\}"
    )

    test_pattern = re.compile(
        r"test: \{'epoch': (\d+), 'time_epoch': ([\d.]+), 'loss': ([\d.]+), 'lr': ([\d.]+), 'params': (\d+), 'time_iter': ([\d.]+), 'accuracy': ([\d.]+), 'auc': ([\d.]+), 'ap': ([\d.]+)\}"
    )
    with open(filename, 'r') as f:
        content = f.readlines()

    # Prepare lists to store extracted data
    data = []
    epoch_data = {'epoch': None}
    param_pairs = []  # Track pairs of MPNN and Global Attention parameters for each layer
    train_found = False
    val_found = False
    test_found = False
    # Process the lines one by one
    for line in content:
        # Match the train, val, and test patterns
        if train_pattern.match(line):
            train_found = True
            match = train_pattern.search(line)
            epoch, time_epoch, eta, eta_hours, loss, lr, params, time_iter, accuracy, auc, ap = match.groups()
            epoch_data.update({
                'epoch': int(epoch),
                'test_time_epoch': float(time_epoch),
                'train_eta': float(eta),
                'train_eta_hours': float(eta_hours),
                'test_loss': float(loss),
                'test_lr': float(lr),
                'test_params': int(params),
                'test_time_iter': float(time_iter),
                'test_accuracy': float(accuracy),
                'test_auc': float(auc),
                'test_ap': float(ap),
            })

        elif val_pattern.match(line):
            val_found = True
            match = val_pattern.search(line)
            epoch, time_epoch, loss, lr, params, time_iter, accuracy, auc, ap = match.groups()
            epoch_data.update({
                'epoch': int(epoch),
                'val_time_epoch': float(time_epoch),
                'val_loss': float(loss),
                'val_lr': float(lr),
                'val_params': int(params),
                'val_time_iter': float(time_iter),
                'val_accuracy': float(accuracy),
                'val_auc': float(auc),
                'val_ap': float(ap)
            })

        elif test_pattern.match(line):
            test_found = True
            match = test_pattern.search(line)
            epoch, time_epoch, loss, lr, params, time_iter, accuracy, auc, ap = match.groups()
            epoch_data.update({
                'epoch': int(epoch),
                'test_time_epoch': float(time_epoch),
                'test_loss': float(loss),
                'test_lr': float(lr),
                'test_params': int(params),
                'test_time_iter': float(time_iter),
                'test_accuracy': float(accuracy),
                'test_auc': float(auc),
                'test_ap': float(ap)
            })
        
        param_match = param_pattern.search(line)
        if param_match:
            mpnn_value = float(param_match.group(1))
            global_attention_value = float(param_match.group(2))
            param_pairs.append({'mpnn_param': mpnn_value, 'global_attention_param': global_attention_value})
        
        if val_found and train_found and test_found and len(param_pairs) == 5:
            # Flatten param pairs into columns for the DataFrame
            for i, params in enumerate(param_pairs):
                epoch_data[f'mpnn_param_layer_{i+1}'] = params['mpnn_param']
                epoch_data[f'global_attention_param_layer_{i+1}'] = params['global_attention_param']

            data.append(epoch_data.copy())
            epoch_data = {'epoch': None}
            param_pairs = []
            train_found = False
            val_found = False
            test_found = False
            

    df = pd.DataFrame(data)
    return df
    
# Ensure the output directory exists
output_dir = '/Users/michaelbest/Desktop/CombineGraphGPS/results_michael/analyze_results'
input_dir = '/Users/michaelbest/Desktop/CombineGraphGPS/results_michael'


# Loop through each file that matches the name pattern
for filename in os.listdir(input_dir):
    if filename.endswith('out'):
        print(filename)
        file = os.path.join(input_dir,filename)  # Current file to process
        csv_filename = os.path.join(output_dir, f"{filename.split('.')[0]}.csv")
        if filename.startswith('molhiv'):
        # Parse data and save to CSV
            data = parse_output_hiv(file)
            data.to_csv(csv_filename)
        if filename.startswith('molpcba'):
            data = parse_output_pcba(file)
            data.to_csv(csv_filename)
        print(f"Processed {filename} and saved to {csv_filename}")
