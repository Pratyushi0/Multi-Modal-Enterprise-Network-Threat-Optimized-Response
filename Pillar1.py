import pandas as pd
import numpy as np

def run_contextual_ewma(matrix_path):
    df = pd.read_csv(matrix_path).sort_values(by=['user', 'day'])
    
    # Compute user-specific adaptive rolling context (14-event window span)
    span = 14
    df['ewm_mean_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=span).mean())
    df['ewm_std_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=span).std()).fillna(1.0)
    
    df['ewm_mean_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=span).mean())
    df['ewm_std_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=span).std()).fillna(1.0)
    
    # Transform raw data spaces to Z-Scores relative purely to individual histories
    df['login_deviation'] = (df['total_logins'] - df['ewm_mean_logins']) / df['ewm_std_logins']
    df['usb_deviation'] = (df['usb_file_copies'] - df['ewm_mean_usb']) / df['ewm_std_usb']
    
    df.fillna(0, inplace=True)
    return df