## get the best epoch results of each model
## get the mpnn and global attention results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
best_ep ={'molhiv_GPS': 34, 'molhiv_GPS+RWSE': 12, 'molhiv_GPS+RWSEdev': 94, 'molpcba_GPS': 55, 'zinc': 1577}

def add_best_epoch():
    # add best epoch column to each csv§
    # add row at the end of the dataframe with all the best epoch results taken from the csv given that the csv is sirted by epoch number
    for file in os.listdir('results_michael/analyze_results/'):
        if file.endswith('.csv'):
            df = pd.read_csv('results_michael/analyze_results/' + file)
            df['best_epoch'] = best_ep[file.split('.')[0]]
            df.to_csv('results_michael/analyze_results/' + file, index=False)
            print('results_michael/analyze_results/' + file)

def delete_last_row():
    for file in os.listdir('results_michael/analyze_results/'):
        if file.endswith('.csv'):
            df = pd.read_csv('results_michael/analyze_results/' + file)
            df = df[:-1]
            df.to_csv('results_michael/analyze_results/' + file, index=False)

    

# i want to create graphs of the training and validation loss for each model

def plot_loss(csv_file):
    df = pd.read_csv(csv_file)
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    if 'test_loss' in df.columns:
        plt.plot(df['epoch'], df['test_loss'], label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for ' + csv_file.split('/')[-1].split('.')[0])
    plt.legend()
    plt.savefig('results_michael/analyze_results/plots/' + csv_file.split('/')[-1].split('.')[0] + '_loss.png')
    plt.show()

def plot_accuracy(csv_file):
    df = pd.read_csv(csv_file)
    if 'train_accuracy'  in df.columns:
        plt.plot(df['epoch'], df['train_accuracy'], label='train_accuracy')
    if 'val_accuracy' in df.columns:
        plt.plot(df['epoch'], df['val_accuracy'], label='val_accuracy')
    if 'test_accuracy' in df.columns:
        plt.plot(df['epoch'], df['test_accuracy'], label='test_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for ' + csv_file.split('/')[-1].split('.')[0])
    plt.legend()
    plt.savefig('results_michael/analyze_results/plots/' + csv_file.split('/')[-1].split('.')[0] + '_accuracy.png')
    plt.show()
    
def plot_layer_parameters(csv_file):
    df = pd.read_csv(csv_file)
    
    # Initialize the figure for plotting
    plt.figure(figsize=(8, 6))

    # Use regular expressions to find columns that match the pattern
    mpnn_columns = [col for col in df.columns if re.match(r"mpnn_param_layer_\d+", col)]
    global_attention_columns = [col for col in df.columns if re.match(r"global_attention_param_layer_\d+", col)]
    
    max_abs = 0
    for mpnn_col, global_attention_col in zip(mpnn_columns, global_attention_columns):
        diff = df[mpnn_col] - df[global_attention_col]
        plt.plot(df['epoch'], diff, label=f"layer {mpnn_col.split('_')[-1]}")
        if np.max(np.abs(diff)) > max_abs:
            max_abs = np.max(np.abs(diff))
        
    # fix the horizontal line at 0 and upper/under line in -2.5/+2.5
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(max_abs, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(-max_abs, color='black', linestyle='--', linewidth=0.5)
    plt.text(len(diff) / 2, max_abs+0.1, "MPNN", fontsize=10, va='center', ha='center', color='blue')
    plt.text(len(diff) / 2, -max_abs-0.1, "Global Attention", fontsize=10, va='center', ha='center', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Difference in Parameters')
    
    # Add labels and title
    plt.title('MPNN to Global Attention Parameters for ' + csv_file.split('/')[-1].split('.')[0])
    plt.legend()
    plot_filename = f"plots/layer_parameters_plot.png"
    plt.savefig('results_michael/analyze_results/plots/' + csv_file.split('/')[-1].split('.')[0] + '_layer_parameters.png')
    plt.show()
    print(f"Plot saved to {plot_filename}")


def plot_all():
    for file in os.listdir('results_michael/analyze_results/'):
        if file.endswith('.csv') and file not in ['pcqm4m_GPS.csv', 'pcqm4m_GPS+RWSE.csv']:
            pd.read_csv('results_michael/analyze_results/' + file)
            plot_loss('results_michael/analyze_results/' + file)
            plot_accuracy('results_michael/analyze_results/' + file)
            plot_layer_parameters('results_michael/analyze_results/' + file)

def best_results_table():
    # create a table with the best results of each model
    # consider the 
    return None

#plot_all()
#delete_last_row()


    