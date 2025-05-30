import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import re
from glob import glob

#setup
data_dir = 'data_raw'
output_dir = 'data_postproc'
os.makedirs(output_dir, exist_ok=True)

#define functions 
def get_key(status):
    return '2' if status == 'repeat' else 'None'

def compute_accuracy(row):
    status = row['repeat_status']
    resp = row['button_resp.keys']
    if status == 'repeat':
        return int(resp == '2')
    elif status == 'original':
        return int(pd.isna(resp))
    return np.nan

def classify(row):
    status = row['repeat_status']
    resp = row['button_resp.keys']
    if status == 'repeat':
        if resp == '2':
            return 'hit'
        elif pd.isna(resp):
            return 'miss'
        else:
            return 'NaN'
    elif status == 'original':
        if pd.isna(resp):
            return 'correct rejection'
        elif resp == '2':
            return 'false alarm'
        else:
            return 'NaN'
    return 'NaN'

def adjusted_rate(rate, n):
    if rate == 1.0:
        return 1 - 1/(2*n)
    elif rate == 0.0:
        return 1/(2*n)
    return rate

#process each participant file
csv_files = glob(os.path.join(data_dir, '*_N170_*.csv'))
print(f"Found {len(csv_files)} files: {csv_files}")
print(f"Current working directory: {os.getcwd()}")

for file_path in csv_files:
    # Extract participant number
    match = re.search(r'sub[-_]?(\d+)_N170_', os.path.basename(file_path))
    if not match:
        print(f"Skipping file (couldn't extract ID): {file_path}")
        continue
    participant_num = match.group(1)
    #load data and format properly
    df = pd.read_csv(file_path)
    df = df[['name', 'repeat_status', 'stim_type', 'button_resp.rt', 'button_resp.keys']]

    df['correct_key'] = df['repeat_status'].map(get_key)
    df['button_resp.keys'] = df['button_resp.keys'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else np.nan
    )
    df['accuracy'] = df.apply(compute_accuracy, axis=1)
    df['trial_type'] = df.apply(classify, axis=1)

    #signal detection
    hits = (df['trial_type'] == 'hit').sum()
    misses = (df['trial_type'] == 'miss').sum()
    false_alarms = (df['trial_type'] == 'false alarm').sum()
    correct_rejections = (df['trial_type'] == 'correct rejection').sum()

    n_signal = hits + misses
    n_noise = false_alarms + correct_rejections

    if n_signal > 0 and n_noise > 0:
        hit_rate = adjusted_rate(hits / n_signal, n_signal)
        fa_rate = adjusted_rate(false_alarms / n_noise, n_noise)
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    else:
        d_prime = np.nan

    print(f"Participant {participant_num} — d': {d_prime:.3f}")

    #save per-participant results
    out_file = os.path.join(output_dir, f'sub-{participant_num}_N170_postproc.csv')
    df.to_csv(out_file, index=False)
