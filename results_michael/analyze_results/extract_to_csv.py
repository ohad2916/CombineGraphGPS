import os
import re
import pandas as pd

param_pattern = re.compile(
    r"tensor\(\s*\[\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?|[+-]?\d+\.)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?|[+-]?\d+\.)\s*\]\s*,\s*device\s*=\s*'cuda:\s*0'\s*,\s*requires_grad\s*=\s*True\s*\)"
)



def parse_output_hiv(file):
    # Define regex patterns for train, validation, and test metrics, along with parameter values
    # Initialize DataFrame to store results
    
    
    train_pattern = re.compile(
        r"train: \{.*?'epoch': (\d+).*?'time_epoch': ([\de\-.]+),.*?'eta': ([\de\-.]+),.*?'eta_hours': ([\de\-.]+),"
        r".*?'loss': ([\de\-.]+),.*?'lr': ([\de\-.]+),.*?'params': (\d+),.*?'time_iter': ([\de\-.]+),"
        r".*?'accuracy': ([\de\-.]+),.*?'precision': ([\de\-.]+),.*?'recall': ([\de\-.]+),"
        r".*?'f1': ([\de\-.]+),.*?'auc': ([\de\-.]+)"
    )
    val_pattern = re.compile(
        r"val: \{.*?'epoch': (\d+).*?'time_epoch': ([\de\-.]+),.*?'loss': ([\de\-.]+),.*?'lr': ([\de\-.]+),"
        r".*?'params': (\d+),.*?'time_iter': ([\de\-.]+),.*?'accuracy': ([\de\-.]+),"
        r".*?'precision': ([\de\-.]+),.*?'recall': ([\de\-.]+),.*?'f1': ([\de\-.]+),"
        r".*?'auc': ([\de\-.]+)"
    )
    test_pattern = re.compile(
        r"test: \{.*?'epoch': (\d+).*?'time_epoch': ([\de\-.]+),.*?'loss': ([\de\-.]+),.*?'lr': ([\de\-.]+),"
        r".*?'params': (\d+),.*?'time_iter': ([\de\-.]+),.*?'accuracy': ([\de\-.]+),"
        r".*?'precision': ([\de\-.]+),.*?'recall': ([\de\-.]+),.*?'f1': ([\de\-.]+),"
        r".*?'auc': ([\de\-.]+)"
    )

    # Initialize data storage
    data = []
    epoch_data = {'epoch': None}
    param_pairs = []  # Track pairs of MPNN and Global Attention parameters for each layer
    with open(file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            # Extract training metrics
            train_match = train_pattern.search(line)
            if train_match:
 
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
            if 'test_auc' in epoch_data:
                if (len(param_pairs) == 10 and filename in ['molhiv_GPS.out', 'molhiv_GPS+RWSE.out']) or (len(param_pairs) == 2 and filename == 'molhiv_GPS+RWSEdev.out'):
                # Flatten param pairs into columns for the DataFrame
                    for i, params in enumerate(param_pairs):
                        epoch_data[f'mpnn_param_layer_{i+1}'] = params['mpnn_param']
                        epoch_data[f'global_attention_param_layer_{i+1}'] = params['global_attention_param']
                    data.append(epoch_data.copy())
                    epoch_data = {'epoch': None}
                    param_pairs = []
                
        # Convert list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
    return df

# Function to extract data and store it into a single dataframe
def parse_output_pcba(filename):
    # Define regular expression patterns
    train_pattern = re.compile(
        r"train:\s*?\{\s*?'epoch':\s*?(\d+),\s*?'time_epoch':\s*?([\de\-.]+),\s*?'eta':\s*?([\de\-.]+),\s*?'eta_hours':\s*?([\de\-.]+),\s*?'loss':\s*?([\de\-.]+),\s*?'lr':\s*?([\de\-.]+),\s*?'params':\s*?(\d+),\s*?'time_iter':\s*?([\de\-.]+),\s*?'accuracy':\s*?([\de\-.]+),\s*?'auc':\s*?([\de\-.]+),\s*?'ap':\s*?([\de\-.]+)\s*?\}"
    )

    val_pattern = re.compile(
        r"val:\s*?\{\s*?'epoch':\s*?(\d+),\s*?'time_epoch':\s*?([\de\-.]+),\s*?'loss':\s*?([\de\-.]+),\s*?'lr':\s*?([\de\-.]+),\s*?'params':\s*?(\d+),\s*?'time_iter':\s*?([\de\-.]+),\s*?'accuracy':\s*?([\de\-.]+),\s*?'auc':\s*?([\de\-.]+),\s*?'ap':\s*?([\de\-.]+)\s*?\}"
    )

    test_pattern = re.compile(
        r"test:\s*?\{\s*?'epoch':\s*?(\d+),\s*?'time_epoch':\s*?([\de\-.]+),\s*?'loss':\s*?([\de\-.]+),\s*?'lr':\s*?([\de\-.]+),\s*?'params':\s*?(\d+),\s*?'time_iter':\s*?([\de\-.]+),\s*?'accuracy':\s*?([\de\-.]+),\s*?'auc':\s*?([\de\-.]+),\s*?'ap':\s*?([\de\-.]+)\s*?\}"
    )

    
    # Prepare lists to store extracted data
    data = []
    epoch_data = {'epoch': None}
    param_pairs = []  # Track pairs of MPNN and Global Attention parameters for each layer

    # Initialize DataFrame to store results
   
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Process the lines one by one
    for line in lines:
        # Match the train, val, and test patterns
        if train_pattern.match(line):
      
            match = train_pattern.search(line)
            epoch, time_epoch, eta, eta_hours, loss, lr, params, time_iter, accuracy, auc, ap = match.groups()
            
            epoch_data.update({
                'epoch': int(epoch),
                'train_time_epoch': float(time_epoch),
                'train_eta': float(eta),
                'train_eta_hours': float(eta_hours),
                'train_loss': float(loss),
                'train_lr': float(lr),
                'train_params': int(params),
                'train_time_iter': float(time_iter),
                'train_accuracy': float(accuracy),
                'train_auc': float(auc),
                'train_ap': float(ap),
            })

        elif val_pattern.match(line):

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
        
        if 'test_ap' in epoch_data and len(param_pairs) == 5:
                # Flatten param pairs into columns for the DataFrame
                for i, params in enumerate(param_pairs):
                    epoch_data[f'mpnn_param_layer_{i+1}'] = params['mpnn_param']
                    epoch_data[f'global_attention_param_layer_{i+1}'] = params['global_attention_param']
                #print(epoch_data)
                data.append(epoch_data.copy())
                epoch_data = {'epoch': None}
                param_pairs = []
            

    df = pd.DataFrame(data)
    return df
def parse_output_zinc(file):
    # Regular expressions for each data part
    train_pattern = r"train:\s*\{\s*'epoch':\s*(\d+),\s*'time_epoch':\s*([\de\-.]+),\s*'eta':\s*([\de\-.]+),\s*'eta_hours':\s*([\de\-.]+),\s*'loss':\s*([\de\-.]+),\s*'lr':\s*([\de\-.]+),\s*'params':\s*(\d+),\s*'time_iter':\s*([\de\-.]+),\s*'mae':\s*([\de\-.]+),\s*'r2':\s*([\de\-.]+),\s*'spearmanr':\s*([\de\-.]+),\s*'mse':\s*([\de\-.]+),\s*'rmse':\s*([\de\-.]+)\s*\}"
    val_pattern = r"val:\s*\{\s*'epoch':\s*(\d+),\s*'time_epoch':\s*([\de\-.]+),\s*'loss':\s*([\de\-.]+),\s*'lr':\s*([\de\-.]+),\s*'params':\s*(\d+),\s*'time_iter':\s*([\de\-.]+),\s*'mae':\s*([\de\-.]+),\s*'r2':\s*([\de\-.]+),\s*'spearmanr':\s*([\de\-.]+),\s*'mse':\s*([\de\-.]+),\s*'rmse':\s*([\de\-.]+)\s*\}"
    test_pattern = r"test:\s*\{\s*'epoch':\s*(\d+),\s*'time_epoch':\s*([\de\-.]+),\s*'loss':\s*([\de\-.]+),\s*'lr':\s*([\de\-.]+),\s*'params':\s*(\d+),\s*'time_iter':\s*([\de\-.]+),\s*'mae':\s*([\de\-.]+),\s*'r2':\s*([\de\-.]+),\s*'spearmanr':\s*([\de\-.]+),\s*'mse':\s*([\de\-.]+),\s*'rmse':\s*([\de\-.]+)\s*\}"
 #param_pattern = r"Parameter containing:\s*tensor\(\[([\de\-.]+), ([\de\-.]+)\], device='cuda:0', requires_grad=True\)"

    # Initialize DataFrame to store results

    

    # Variables to store the extracted data for each epoch
    epoch_data = {'epoch': None}
    param_pairs = []
    data = []
    found = 0

    # Process the file line by line
    with open(file, "r") as file:
        for line in file.readlines(): 
          
            
            # Check for each pattern
            train_match = re.search(train_pattern, line)
            val_match = re.search(val_pattern, line)
            test_match = re.search(test_pattern, line)
            param_match = re.search(param_pattern, line)
            
            if param_match:
                mpnn_value = float(param_match.group(1))
                global_attention_value = float(param_match.group(2))
                param_pairs.append({'mpnn_param': mpnn_value, 'global_attention_param': global_attention_value})

            # If a training line match is found
            elif train_match:
                # Save training data
                found += 1
                
                epoch_data.update({
                    'epoch': int(train_match.group(1)),
                    'train_epoch_time': float(train_match.group(2)),
                    'train_eta': float(train_match.group(3)),
                    'train_eta_hours': float(train_match.group(4)),
                    'train_loss': float(train_match.group(5)),
                    'train_lr': float(train_match.group(6)),
                    'train_params': int(train_match.group(7)),
                    'train_iter_time': float(train_match.group(8)),
                    'train_mae': float(train_match.group(9)),
                    'train_r2': float(train_match.group(10)),
                    'train_spearmanr': float(train_match.group(11)),
                    'train_mse': float(train_match.group(12)),
                    'train_rmse': float(train_match.group(13))
                    
                })
            
            # If a validation line match is found
            elif val_match:
                found += 1
 
                # Save validation data
                epoch_data.update({
                    #'epoch': int(val_match.group(1)),
                    'val_epoch_time': float(val_match.group(2)),
                    'val_loss': float(val_match.group(3)),
                    'val_lr': float(val_match.group(4)),
                    'val_params': int(val_match.group(5)),
                    'val_iter_time': float(val_match.group(6)),
                    'val_mae': float(val_match.group(7)),
                    'val_r2': float(val_match.group(8)),
                    'val_spearmanr': float(val_match.group(9)),
                    'val_mse': float(val_match.group(10)),
                    'val_rmse': float(val_match.group(11))
                    
                })
            
            # If a test line match is found
            elif test_match:
                # Save test data
                found += 1

                epoch_data.update({
                    #'epoch': int(test_match.group(1)),
                    'test_epoch_time': float(test_match.group(2)),
                    'test_loss': float(test_match.group(3)),
                    'test_lr': float(test_match.group(4)),
                    'test_params': int(test_match.group(5)),
                    'test_iter_time': float(test_match.group(6)),
                    'test_mae': float(test_match.group(7)),
                    'test_r2': float(test_match.group(8)),
                    'test_spearmanr': float(test_match.group(9)),
                    'test_mse': float(test_match.group(10)),
                    'test_rmse': float(test_match.group(11))
                })
           
            # At the end of each epoch (after test line), append data to DataFrame
            if found == 3 and len(param_pairs) == 10:
                    # Add parameter data for each layer
                    for i, params in enumerate(param_pairs):
                        epoch_data[f'mpnn_param_layer_{i+1}'] = params['mpnn_param']
                        epoch_data[f'global_attention_param_layer_{i+1}'] = params['global_attention_param']
                    
                    data.append(epoch_data.copy())
                    epoch_data = {'epoch': None}
                    param_pairs = []
                    found = 0
               
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
        if filename.startswith('zinc'):
            data = parse_output_zinc(file)
            data.to_csv(csv_filename)
        print(f"Processed {filename} and saved to {csv_filename}")
