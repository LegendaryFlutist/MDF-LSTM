import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_lstm_data_standard(df, featCols):
    # 初始化一个空列表来存储所有样本的NumPy数组
    x_list = []
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        # 将DataFrame转换为NumPy数组
        np_data = features_data.values
        # 存储NumPy数组
        # # 标准化
        scaler = StandardScaler()
        np_data = scaler.fit_transform(np_data)
        x_list.append(np_data)

    # 返回序列列表和每个序列的时间步长
    return x_list, [x.shape[0] for x in x_list]


def get_psd_features(df, featCols):
    psd_entropy_list = []
    psd_max_power_list = []
    psd_freq_max_power_list = []
    scaler = MinMaxScaler(feature_range=(0, 1))  # 或者使用 (-1, 1) 取决于您的需求
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        features_data_normalized = scaler.fit_transform(features_data)
        sample_entropy = []
        sample_max_power = []
        sample_freq_max_power = []
        for idx, column in enumerate(featCols):
            # 归一化
            column_data = features_data_normalized[:, idx]
            f, psd = welch(column_data, fs=1.0)  # 计算功率谱密度
            psd_entropy, psd_max_power, psd_freq_max_power = getSpectralFeatures(f, psd)
            sample_entropy.append(psd_entropy)
            sample_max_power.append(psd_max_power)
            sample_freq_max_power.append(psd_freq_max_power)
        psd_entropy_list.append(sample_entropy)
        psd_max_power_list.append(sample_max_power)
        psd_freq_max_power_list.append(sample_freq_max_power)

    psd_entropy_array = np.array(psd_entropy_list)
    psd_max_power_array = np.array(psd_max_power_list)
    psd_freq_max_power_array = np.array(psd_freq_max_power_list)

    return psd_entropy_array, psd_max_power_array, psd_freq_max_power_array


def getSpectralFeatures(f, psd):
    psdMaxPower = max(psd)
    psdMaxPowerIdx = np.argmax(psd)
    psdFreqOfMax = f[psdMaxPowerIdx]

    # 计算整体谱熵
    normPSD = psd / np.sum(psd)
    overallEntropy = -np.sum(normPSD * np.log2(normPSD + 1e-10))  # 添加一个小常数避免对数为负无穷

    return overallEntropy, psdMaxPower, psdFreqOfMax

def get_psd_f(df, featCols):
    psd_mean_list = []
    psd_std_list = []
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        sample_mean = []
        sample_std = []
        for column in features_data.columns:
            f, psd = welch(features_data[column], fs=1.0)
            sample_mean.append(np.mean(psd))
            sample_std.append(np.std(psd))
        psd_mean_list.append(sample_mean)
        psd_std_list.append(sample_std)
    # scaler = MinMaxScaler(feature_range=(0, 1))  # 或者使用 (-1, 1) 取决于您的需求
    # psd_mean_list = scaler.fit_transform(psd_mean_list)
    # psd_std_list = scaler.fit_transform(psd_std_list)
    psd_mean_array = np.array(psd_mean_list)
    psd_std_array = np.array(psd_std_list)
    return psd_mean_array, psd_std_array


def get_lstm_data_min_max(df, featCols):
    # 初始化一个空列表来存储所有样本的NumPy数组
    x_list = []
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        # 将DataFrame转换为NumPy数组
        np_data = features_data.values
        # 存储NumPy数组
        #归一化
        scaler = MinMaxScaler(feature_range=(0, 1))  # 或者使用 (-1, 1) 取决于您的需求
        np_data = scaler.fit_transform(np_data)
        x_list.append(np_data)

    # 返回序列列表和每个序列的时间步长
    return x_list, [x.shape[0] for x in x_list]


def fft_transform(data):
    # 对每一行（样本）进行FFT，返回频域特征
    return np.fft.fft(data)


def get_lstm_data_fft(df, featCols):
    # 初始化一个空列表来存储所有样本的NumPy数组
    x_list = []
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        # 将DataFrame转换为NumPy数组
        np_data = features_data.values
        # 存储NumPy数组
        np_data = fft_transform(np_data)

        x_list.append(np_data)

    # 返回序列列表和每个序列的时间步长
    return x_list, [x.shape[0] for x in x_list]


def get_lstm_data(df, featCols):
    # 初始化一个空列表来存储所有样本的NumPy数组
    x_list = []
    # 遍历所有样本的DataFrame
    for data_df in df['data']:
        # 确保只选择所需的特征列
        features_data = data_df[featCols]
        # 将DataFrame转换为NumPy数组
        np_data = features_data.values
        # 存储NumPy数组
        x_list.append(np_data)

    # 返回序列列表和每个序列的时间步长
    return x_list, [x.shape[0] for x in x_list]
