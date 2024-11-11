## get the best epoch results of each model
## get the mpnn and global attention results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# add best epoch column to each csv§
best_ep_molhiv = 34
best_ep_molhiv_RWSE = 12
best_ep_molhiv_RWSEdev = 94
best_ep_molpcba = 55
# add row at the end of the dataframe with all the best epoch results taken from the csv given that the csv is sirted by epoch number
df = pd.read_csv('results_michael/analyze_results/molhiv_GPS.csv')
best_epoch_row = df[df['epoch'] == best_ep_molhiv]
df = df.append(best_epoch_row)
df.to_csv('results_michael/analyze_results/molhiv_GPS.csv', index=False)

df = pd.read_csv('results_michael/analyze_results/molhiv_GPS+RWSE.csv')
best_epoch_row = df[df['epoch'] == best_ep_molhiv_RWSE]
df = df.append(best_epoch_row)
df.to_csv('results_michael/analyze_results/molhiv_GPS+RWSE.csv', index=False)

df = pd.read_csv('results_michael/analyze_results/molhiv_GPS+RWSEdev.csv')
best_epoch_row = df[df['epoch'] == best_ep_molhiv_RWSEdev]
df = df.append(best_epoch_row)
df.to_csv('results_michael/analyze_results/molhiv_GPS+RWSEdev.csv', index=False)

df = pd.read_csv('results_michael/analyze_results/molpcba_GPS.csv')
best_epoch_row = df[df['epoch'] == best_ep_molpcba]
df = df.append(best_epoch_row)
df.to_csv('results_michael/analyze_results/molpcba_GPS.csv', index=False)





    