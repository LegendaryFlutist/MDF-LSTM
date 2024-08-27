import os

import pandas as pd

def down_sampling_v2(eye_data, eda_data, emg_data, resp_data, ecg_data):
    sensor_freq_dict = {
        'eye': 504,
        'eda': 2048,
        'emg': 2048,
        'resp': 256,
        'ace': 256,
        'ecg': 256
    }

    sensor_freq_dict1 = {
        'eye': 252,
        'eda': 1024,
        'emg': 128,
        'resp': 128,
        'ace': 128,
        'ecg': 128
    }
    sensor_freq_dict05 = {
        'eye': 126,
        'eda': 512,
        'emg': 64,
        'resp': 64,
        'ace': 64,
        'ecg': 64
    }
    sensor_freq_dict025 = {
        'eye': 63,
        'eda': 256,
        'emg': 32,
        'resp': 32,
        'ace': 32,
        'ecg': 32
    }

    max_start_time = max(eye_data['time_dn'][0], eda_data['time_dn'][0], emg_data['time_dn'][0],
                         resp_data['time_dn'][0],
                         ecg_data['time_dn'][0], )
    min_end_time = min(eye_data['time_dn'].iloc[-1], eda_data['time_dn'].iloc[-1], emg_data['time_dn'].iloc[-1],
                       resp_data['time_dn'].iloc[-1],
                       ecg_data['time_dn'].iloc[-1],
            )
    eye_data = eye_data.loc[(eye_data['time_dn'] >= max_start_time) & (eye_data['time_dn'] <= min_end_time)]
    eda_data = eda_data.loc[(eda_data['time_dn'] >= max_start_time) & (eda_data['time_dn'] <= min_end_time)]
    emg_data = emg_data.loc[(emg_data['time_dn'] >= max_start_time) & (emg_data['time_dn'] <= min_end_time)]
    resp_data = resp_data.loc[(resp_data['time_dn'] >= max_start_time) & (resp_data['time_dn'] <= min_end_time)]
    ecg_data = ecg_data.loc[(ecg_data['time_dn'] >= max_start_time) & (ecg_data['time_dn'] <= min_end_time)]


    result_df = pd.DataFrame(columns=[])
    eye_num, eda_num, emg_num, resp_num, ace_num, ecg_num = 0, 0, 0, 0, 0, 0
    while eye_num < len(eye_data) and eda_num < len(eda_data) and emg_num < len(emg_data) and ecg_num < len(
            ecg_data) and resp_num < len(resp_data):
        eye_sub = eye_data.iloc[eye_num:eye_num + sensor_freq_dict.get('eye')]
        eda_sub = eda_data.iloc[eda_num:eda_num + sensor_freq_dict.get('eda')]
        emg_sub = emg_data.iloc[emg_num:emg_num + sensor_freq_dict.get('emg')]
        resp_sub = resp_data.iloc[resp_num:resp_num + sensor_freq_dict.get('resp')]
        ecg_sub = ecg_data.iloc[ecg_num:ecg_num + sensor_freq_dict.get('ecg')]

        # 计算均值（排除time_dn字段）
        eye_mean = eye_sub.drop(columns=['time_dn']).mean()
        eda_mean = eda_sub.drop(columns=['time_dn']).mean()
        emg_mean = emg_sub.drop(columns=['time_dn']).mean()
        resp_mean = resp_sub.drop(columns=['time_dn']).mean()
        ecg_mean = ecg_sub.drop(columns=['time_dn']).mean()
        mean_row = pd.concat(
            [pd.DataFrame([eye_mean]), pd.DataFrame([emg_mean]), pd.DataFrame([resp_mean]), pd.DataFrame([ecg_mean]),
             pd.DataFrame([eda_mean])], axis=1)
        result_df = pd.concat([mean_row, result_df], axis=0, ignore_index=True)
        eye_num = eye_num + sensor_freq_dict.get('eye')
        eda_num = eda_num + sensor_freq_dict.get('eda')
        emg_num = emg_num + sensor_freq_dict.get('emg')
        resp_num = resp_num + sensor_freq_dict.get('resp')
        ecg_num = ecg_num + sensor_freq_dict.get('ecg')
    return result_df


def down_sampling(eye_data, eda_data, emg_data, resp_data, ecg_data, ace_data):
    sensor_freq_dict = {
        'eye': 252,
        'eda': 1024,
        'emg': 128,
        'resp': 128,
        'ace': 128,
        'ecg': 128
    }

    max_start_time = max(eye_data['time_dn'][0], eda_data['time_dn'][0], emg_data['time_dn'][0],
                         resp_data['time_dn'][0],
                         ecg_data['time_dn'][0], ace_data['time_dn'][0])
    min_end_time = min(eye_data['time_dn'].iloc[-1], eda_data['time_dn'].iloc[-1], emg_data['time_dn'].iloc[-1],
                       resp_data['time_dn'].iloc[-1],
                       ecg_data['time_dn'].iloc[-1],
                       ace_data['time_dn'].iloc[-1])
    eye_data = eye_data.loc[(eye_data['time_dn'] >= max_start_time) & (eye_data['time_dn'] <= min_end_time)]
    eda_data = eda_data.loc[(eda_data['time_dn'] >= max_start_time) & (eda_data['time_dn'] <= min_end_time)]
    emg_data = emg_data.loc[(emg_data['time_dn'] >= max_start_time) & (emg_data['time_dn'] <= min_end_time)]
    resp_data = resp_data.loc[(resp_data['time_dn'] >= max_start_time) & (resp_data['time_dn'] <= min_end_time)]
    ecg_data = ecg_data.loc[(ecg_data['time_dn'] >= max_start_time) & (ecg_data['time_dn'] <= min_end_time)]
    ace_data = ace_data.loc[(ace_data['time_dn'] >= max_start_time) & (ace_data['time_dn'] <= min_end_time)]

    result_df = pd.DataFrame(columns=[])
    eye_num, eda_num, emg_num, resp_num, ace_num, ecg_num = 0, 0, 0, 0, 0, 0
    while eye_num < len(eye_data) and eda_num < len(eda_data) and emg_num < len(emg_data) and ecg_num < len(
            ecg_data) and resp_num < len(resp_data) and ace_num < len(ace_data):
        eye_sub = eye_data.iloc[eye_num:eye_num + sensor_freq_dict.get('eye')]
        eda_sub = eda_data.iloc[eda_num:eda_num + sensor_freq_dict.get('eda')]
        emg_sub = emg_data.iloc[emg_num:emg_num + sensor_freq_dict.get('emg')]
        resp_sub = resp_data.iloc[resp_num:resp_num + sensor_freq_dict.get('resp')]
        ecg_sub = ecg_data.iloc[ecg_num:ecg_num + sensor_freq_dict.get('ecg')]
        ace_sub = ace_data.iloc[ace_num:ace_num + sensor_freq_dict.get('ace')]
        # 计算均值（排除time_dn字段）
        eye_mean = eye_sub.drop(columns=['time_dn']).mean()
        eda_mean = eda_sub.drop(columns=['time_dn']).mean()
        emg_mean = emg_sub.drop(columns=['time_dn']).mean()
        resp_mean = resp_sub.drop(columns=['time_dn']).mean()
        ecg_mean = ecg_sub.drop(columns=['time_dn']).mean()
        ace_mean = ace_sub.drop(columns=['time_dn']).mean()
        mean_row = pd.concat(
            [pd.DataFrame([eye_mean]), pd.DataFrame([emg_mean]), pd.DataFrame([resp_mean]), pd.DataFrame([ecg_mean]),
             pd.DataFrame([eda_mean]), pd.DataFrame([ace_mean])], axis=1)
        result_df = pd.concat([mean_row, result_df], axis=0, ignore_index=True)
        eye_num = eye_num + sensor_freq_dict.get('eye')
        eda_num = eda_num + sensor_freq_dict.get('eda')
        emg_num = emg_num + sensor_freq_dict.get('emg')
        resp_num = resp_num + sensor_freq_dict.get('resp')
        ace_num = ace_num + sensor_freq_dict.get('ace')
        ecg_num = ecg_num + sensor_freq_dict.get('ecg')

    return result_df