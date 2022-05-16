import os
import numpy as np
import pandas as pd
import torch
import random
import re
import json
from pathlib import PosixPath,WindowsPath
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import glob
from pdb import set_trace
# DATA_DIRS = '/mnt/infonas/data/pratham/Forecasting/DILATE'
DATA_DIRS = '/mnt/cat/data/sandy/Forecasting/'
# DATA_DIRS = '.'
def generate_train_dev_test_data(data, N_input):
    train_per = 0.6
    dev_per = 0.2
    N = len(data)

    data_train = data[:int(train_per*N)]
    data_dev = data[int(train_per*N)-N_input:int((train_per+dev_per)*N)]
    data_test = data[int((train_per+dev_per)*N)-N_input:]

    return  (data_train, data_dev, data_test)

def create_forecast_io_seqs(data, enc_len, dec_len, stride):

    data_in, data_out = [], []
    for idx in range(0, len(data), stride):
        if idx+enc_len+dec_len <= len(data):
            data_in.append(data[idx:idx+enc_len])
            data_out.append(data[idx+enc_len:idx+enc_len+dec_len])

    data_in = np.array(data_in)
    data_out = np.array(data_out)
    return data_in, data_out


def process_start_string(start_string, freq):
    '''
    Source: 
    https://github.com/mbohlkeschneider/gluon-ts/blob/442bd4ffffa4a0fcf9ae7aa25db9632fbe58a7ea/src/gluonts/dataset/common.py#L306
    '''

    timestamp = pd.Timestamp(start_string, freq=freq)
    # 'W-SUN' is the standardized freqstr for W
    if timestamp.freq.name in ("M", "W-SUN"):
        offset = to_offset(freq)
        timestamp = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0, nanosecond=0
        )
        return pd.Timestamp(
            offset.rollback(timestamp), freq=offset.freqstr
        )
    if timestamp.freq == 'B':
        # does not floor on business day as it is not allowed
        return timestamp
    return pd.Timestamp(
        timestamp.floor(timestamp.freq), freq=timestamp.freq
    )

def shift_timestamp(ts, offset):
    result = ts + offset * ts.freq
    return pd.Timestamp(result, freq=ts.freq)

def get_date_range(start_string, freq, seq_len):
    start = process_start_string(start_string, freq)
    end = shift_timestamp(start, seq_len)
    full_date_range = pd.date_range(start, end, freq=freq)
    return full_date_range


def get_list_of_dict_format(data,inject,mask):
    data_new = list()
    for entry,inj,m in zip(data,inject,mask):
        entry_dict = dict()
        entry_dict['target'] = entry
        entry_dict['target_inj']=inj
        entry_dict['target_mask']=m
        data_new.append(entry_dict)
    return data_new

def prune_dev_test_sequence(data, seq_len):
    for i in range(len(data)):
        
        data[i]['target'] = data[i]['target'][-seq_len:]
        data[i]['target_inj'] = data[i]['target_inj'][-seq_len:]
        data[i]['target_mask'] = data[i]['target_mask'][-seq_len:]
        data[i]['feats'] = data[i]['feats'][-seq_len:]
    return data

def prune_dev_test(data, seq_len):
    for i in range(len(data)):
        
        data[i]['target'] = data[i]['target'][-seq_len:]
        data[i]['feats'] = data[i]['feats'][-seq_len:]
    return data

def parse_smd(dataset_name, N_input, N_output, t2v_type=None):
    path = PosixPath(os.path.join(DATA_DIRS,'data', 'ServerMachineDataset')).expanduser()
    bmk_dirs = ["train", "test", "test_label"]
    train_files = [fn for fn in os.listdir(path / "train") if fn.endswith(".txt")]
    train_files.sort()
    train_dataset, test_dataset, test_labels, interpretation_labels = [], [], [], []
    # for fn_i in tqdm.tqdm(train_files):
    for fn_i in train_files[:1]: # TODO Selecting 5 series is temp, change this
        ts_id = re.sub(".txt$", "", fn_i)
        # print(fn_i)
        ts_train_np, ts_test_np, test_anomalies = [
            np.genfromtxt(
                fname=path / dir_j / fn_i,
                dtype=np.float32,
                delimiter=",",
            )
            for dir_j in bmk_dirs
        ]
        train_anomalies = None

        with open(path / 'interpretation_label' / fn_i , 'r') as f:
            interpretation_label = []
            for line in f:
                indices = line.split(':')[-1].strip().split(',')
                indices = [int(idx) for idx in indices]
                interpretation_label.append(indices)

        train_dataset.append(ts_train_np)
        test_dataset.append(ts_test_np)
        test_labels.append(test_anomalies)
        interpretation_labels.append(interpretation_label)

    data = []
    for i, (trn_seq, test_seq, test_lbl) in enumerate(zip(train_dataset, test_dataset, test_labels)):
        # assert len(trn_seq) == len(test_seq)
        # series = dict()
        # series['target'] = np.concatenate([trn_seq, test_seq], axis=0)
        # series['target'] = torch.tensor(series['target'], dtype=torch.float)
        # train_lbl = np.zeros((len(trn_seq), 1), dtype=float)
        # test_lbl = np.expand_dims(test_lbl, axis=-1)
        # series['label'] = np.concatenate([train_lbl, test_lbl], axis=0)
        # series['label'] = torch.tensor(series['label'], dtype=torch.float)
        # series['ts_id'] = i * torch.ones(len(series['target']))
        series = np.concatenate([trn_seq, test_seq], axis=0)
        
        data.append(series[...,0])
    # set_trace()
    n = len(data[0])
    train_data_len = train_dataset[0].shape[0]
    units = train_data_len//N_output
    dev_len = int(0.2*units) * N_output
    test_len =  ((n-train_data_len)//N_output)*N_output
    train_len = n - dev_len - test_len
    
    feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats = np.expand_dims(feats_date, axis=0)
    data_mask = np.zeros_like(data)

    data = torch.tensor(data, dtype=torch.float)
    data_inj = torch.tensor(data, dtype=torch.float)
    data_mask = torch.tensor(data_mask, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)
    data_train = data[:, :train_len]
    data_inj_train = data_inj[:, :train_len]
    data_mask_train = data_mask[:, :train_len]
    feats_train = feats[:, :train_len]
    data_dev, data_test,data_inj_dev,data_inj_test,data_mask_dev,data_mask_test = [], [],[],[],[],[]
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    seq_len = 2*N_input+N_output  #(336*2 + 168 = 840)
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_inj_dev.append(data_inj[i,:j])
                data_mask_dev.append(data_mask[i,:j])
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_inj_test.append(torch.tensor(data_inj[i,:j]))
                mask = torch.zeros_like(data_mask[i,:j])
                data_mask_test.append(mask)
                # set_trace()
                data_test.append(torch.tensor(data[i, :j]))
                feats_test.append(torch.tensor(feats[i, :j]))
                test_tsid_map.append(torch.tensor(i))
        
    
    data_train = get_list_of_dict_format(data_train,data_inj_train,data_mask_train)
    data_dev = get_list_of_dict_format(data_dev,data_inj_dev,data_mask_dev)
    data_test = get_list_of_dict_format(data_test,data_inj_test,data_mask_test)


    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]
    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    data_dev = prune_dev_test_sequence(data_dev, seq_len) #54 * 840
    data_test = prune_dev_test_sequence(data_test, seq_len)
    # set_trace()
    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )






def parse_gecco(dataset_name, N_input, N_output, t2v_type=None):

#    train_len = 52*168
#    dev_len = 17*168
#    test_len = 17*168
#    n = train_len + dev_len + test_len
#    df = pd.read_csv('../Informer2020/data/ETT/ETTh1.csv').iloc[:n]

    df = pd.read_csv(DATA_DIRS+'data/water_quality/gecco2018.csv')
  
    df = df[40000:120000]

    data = df[['pH']].to_numpy().T
    df['EVENT'] = df['EVENT'].map({False:0, True: 1})
    data_mask = df[['EVENT']].to_numpy().T
    # set_trace()
    #data = np.expand_dims(data, axis=-1)

    # data_mask = np.zeros_like(data,dtype=float)
    n = data.shape[1]
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len
    # set_trace()
    #train_len = int(0.6*n)
    #dev_len = int(0.2*n)
    #test_len = n - train_len - dev_len

    # feats_cont = np.expand_dims(df[['HUFL','HULL','MUFL','MULL','LUFL','LULL']].to_numpy(), axis=0)

    cal_date = pd.to_datetime(df['Time'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
                cal_date.dt.minute
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    feats_min = np.expand_dims(np.expand_dims(cal_date.dt.minute.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    #feats = np.concatenate([feats_discrete, feats_cont], axis=-1)
    #feats = feats_discrete
    feats = np.concatenate([feats_hod, feats_min, feats_date], axis=-1)

    #data = (data - np.mean(data, axis=0, keepdims=True)).T

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)
    data_mask = torch.tensor(data_mask, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]
    data_mask_train = data_mask[:, :train_len]
    data_dev, data_test,data_inj_dev,data_inj_test,data_mask_dev,data_mask_test = [], [],[],[],[],[]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    
    seq_len = 2*N_input+N_output
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                data_mask_dev.append(data_mask[i,:j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output-seq_len, n+1, N_output):
            if j <= n:
                # print(i,j,n)
                data_test.append(data[i, :j])
                data_mask_test.append(data_mask[i,:j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)
                for k in range(0,N_input-25,25):
                    mask = torch.zeros_like(data_mask[i,:j])
                    
                    start = j-seq_len+N_input
                    # print(start+k,start+k+50)
                    mask[start+k:start+k+50]=1
                    data_mask_test.append(mask)
                    # set_trace()
                    data_test.append(torch.tensor(data[i, :j]))
                    feats_test.append(torch.tensor(feats[i, :j]))
                    test_tsid_map.append(torch.tensor(i))
    


    data_train = get_list_of_dict_format(data_train,data_train,data_mask_train)
    data_dev = get_list_of_dict_format(data_dev,data_dev,data_mask_dev)
    data_test = get_list_of_dict_format(data_test,data_test,data_mask_test)


    decompose_type = 'STL'
    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    # feats_info = {0:(24, 16), 1:(60, 16), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1), 6:(0, 1)}
    feats_info = {0:(24, 1),1:(60,16)}
    # feats_info = {0:(0, 1)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)
    # import ipdb ; ipdb.set_trace()

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )


def parse_energy_data(dataset_name, N_input, N_output, t2v_type=None):

#    train_len = 52*168
#    dev_len = 17*168
#    test_len = 17*168
#    n = train_len + dev_len + test_len
#    df = pd.read_csv('../Informer2020/data/ETT/ETTh1.csv').iloc[:n]

    df = pd.read_csv(DATA_DIRS+'data/energy-anomaly-detection/train.csv')
    
    df = df[df['building_id']==966].interpolate(limit_direction='both',method='linear')
    data = df[['meter_reading']].to_numpy().T
    # set_trace()
    data_mask = df[['anomaly']].to_numpy().T
    #data = np.expand_dims(data, axis=-1)
    # test_data = np.load(os.path.join(DATA_DIRS,"data","water_quality","gecco_test_mask.npy"))
    # test_l = len(test_data)
    # data_mask = np.zeros_like(data,dtype=float)
    n = data.shape[1]
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    ### generated masking
    # data_mask[...,-test_l-N_output:-N_output] = test_data 
  
    # feats_cont = np.expand_dims(df[['HUFL','HULL','MUFL','MULL','LUFL','LULL']].to_numpy(), axis=0)

    cal_date = pd.to_datetime(df['timestamp'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    # feats_min = np.expand_dims(np.expand_dims(cal_date.dt.minute.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    #feats = np.concatenate([feats_discrete, feats_cont], axis=-1)
    #feats = feats_discrete
    feats = np.concatenate([feats_hod, feats_date], axis=-1)

    #data = (data - np.mean(data, axis=0, keepdims=True)).T

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)
    data_mask = torch.tensor(data_mask, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]
    data_mask_train = data_mask[:, :train_len]
    data_dev, data_test,data_inj_dev,data_inj_test,data_mask_dev,data_mask_test = [], [],[],[],[],[]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    seq_len = 2*N_input+N_output
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                data_mask_dev.append(data_mask[i,:j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output-seq_len, n+1, N_output):
            if j <= n:
                # print(i,j,n)
                data_test.append(data[i, :j])
                data_mask_test.append(data_mask[i,:j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)
                for k in range(0,N_input-25,25):
                    mask = torch.zeros_like(data_mask[i,:j])
                    
                    start = j-seq_len+N_input
                    # print(start+k,start+k+50)
                    mask[start+k:start+k+50]=1
                    data_mask_test.append(mask)
                    # set_trace()
                    data_test.append(torch.tensor(data[i, :j]))
                    feats_test.append(torch.tensor(feats[i, :j]))
                    test_tsid_map.append(torch.tensor(i))
    

    data_train = get_list_of_dict_format(data_train,data_train,data_mask_train)
    data_dev = get_list_of_dict_format(data_dev,data_dev,data_mask_dev)
    data_test = get_list_of_dict_format(data_test,data_test,data_mask_test)


    decompose_type = 'STL'
    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    # feats_info = {0:(24, 16), 1:(60, 16), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1), 6:(0, 1)}
    feats_info = {0:(24, 1),1:(31,16)}
    # feats_info = {0:(0, 1)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)
    # import ipdb ; ipdb.set_trace()

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )



def parse_electricity(dataset_name, N_input, N_output, t2v_type=None):
    from pdb import set_trace
    #df = pd.read_csv('data/electricity_load_forecasting_panama/continuous_dataset.csv')
    df = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'continuous_dataset.csv')
    )
    df_inject   = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', '2_percent_electricity.csv')
    )
    # df_mask   = pd.read_csv(
    #     os.path.join('.', 'data', 'masked.csv')
    # )


    # test_data = np.load(os.path.join(DATA_DIRS,"Outliers","Outlier","test_data.npy"))
    # test_l = len(test_data)
    data = df[['nat_demand']].to_numpy().T
    data_inj = df_inject[['nat_demand']].to_numpy().T
    data_mask = df_inject[['label']].to_numpy().T
    
    # data_inj = data
    #n = data.shape[1]
    n = (1903 + 1) * 24 # Select first n=1904*24 entries because of non-stationarity in the data after first n values
    data = data[:, :n]
    data_inj = data_inj[:, :n]
    data_mask = data_mask[:, :n]
    # data_inj[...,-test_l:] = test_data 
    df = df.iloc[:n]

    # set_trace()
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    #import ipdb ; ipdb.set_trace()

    cal_date = pd.to_datetime(df['datetime'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    feats = np.concatenate([feats_hod, feats_date], axis=-1)

    data = torch.tensor(data, dtype=torch.float)
    data_inj = torch.tensor(data_inj, dtype=torch.float)
    data_mask = torch.tensor(data_mask, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    data_inj_train = data_inj[:, :train_len]
    data_mask_train = data_mask[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test,data_inj_dev,data_inj_test,data_mask_dev,data_mask_test = [], [],[],[],[],[]
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    seq_len = 2*N_input+N_output  #(336*2 + 168 = 840)
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_inj_dev.append(data_inj[i,:j])
                data_mask_dev.append(data_mask[i,:j])
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_inj_test.append(torch.tensor(data_inj[i,:j]))
                mask = torch.zeros_like(data_mask[i,:j])
                data_mask_test.append(mask)
                # set_trace()
                data_test.append(torch.tensor(data[i, :j]))
                feats_test.append(torch.tensor(feats[i, :j]))
                test_tsid_map.append(torch.tensor(i))
                for k in range(0,N_input-25,25):
                    data_inj_test.append(torch.tensor(data_inj[i,:j]))
                    mask = torch.zeros_like(data_mask[i,:j])
                    
                    start = j-seq_len+N_input
                    # print(start+k,start+k+50)
                    mask[start+k:start+k+50]=1
                    data_mask_test.append(mask)
                    # set_trace()
                    data_test.append(torch.tensor(data[i, :j]))
                    feats_test.append(torch.tensor(feats[i, :j]))
                    test_tsid_map.append(torch.tensor(i))
    
    
    data_train = get_list_of_dict_format(data_train,data_inj_train,data_mask_train)
    data_dev = get_list_of_dict_format(data_dev,data_inj_dev,data_mask_dev)
    data_test = get_list_of_dict_format(data_test,data_inj_test,data_mask_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]
    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    data_dev = prune_dev_test_sequence(data_dev, seq_len) #54 * 840
    data_test = prune_dev_test_sequence(data_test, seq_len)
    # set_trace()
    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )
