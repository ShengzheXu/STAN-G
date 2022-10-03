"""
"""

from tracemalloc import Snapshot
import numpy as np
import pickle
import pandas as pd

from datetime import datetime

import os
from glob import glob
from flowattacker.model.context_loader import BehaviorContextLoader
from flowattacker.helper.netflow_helper import *

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline

class NetflowProcessor():
    '''
        columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
        raw_columns = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'pkt', 'byt', 'lable']
    '''
    
    def __init__(self, raw_columns=None, individual_user=False, working_dir='tmp') -> None:
        self.starting_time_stamp = None
    
        self.column_cont = None
        self.column_disc = None
        self.preprocessor = None
        self.raw_columns = raw_columns if raw_columns else ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'pkt', 'byt', 'lable']
        self.individual_user=individual_user
        self.working_dir = working_dir

    def save_processor(self):
        os.makedirs(os.path.join(os.getcwd(), self.working_dir, 'nn_input'), exist_ok=True)
        filename = os.path.join(os.getcwd(), self.working_dir, 'nn_input', 'netflow_processor.pickle')
        with open(filename, 'wb') as outp: 
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def load_processor(self):
        filename = os.path.join(os.getcwd(), self.working_dir, 'nn_input', 'netflow_processor.pickle')
        with open(filename, 'rb') as inp:
            self = pickle.load(inp)
        return self
    
    def get_col_by_name(self, attr):
        if attr in self.column_disc:
            idx = self.column_disc.index(attr)
            return idx, self.disc_preprocessor.categories_[idx]
        else:
            idx = self.column_cont.index(attr)+ len(self.column_disc)
            return idx, None

    def corpus_wise_norm_fit(self, corpus_df: list, cont_scaler: str = 's'):
        corpus_whole_view = pd.concat(corpus_df)
        
        numeric_features = self.column_cont
        categorical_features = self.column_disc
        cat_unique = corpus_whole_view[categorical_features].nunique()
        self.column_disc_num = [cat_unique[col] for col in self.column_disc]

        
        self.cont_preprocessor = RobustScaler() if cont_scaler == 'r' else StandardScaler()
        self.disc_preprocessor = OrdinalEncoder()

        self.cont_preprocessor.fit(corpus_whole_view[self.column_cont])
        self.disc_preprocessor.fit(corpus_whole_view[self.column_disc])
        print(self.disc_preprocessor.categories_[0])
        print(self.disc_preprocessor.categories_[1])

        # cont_scaler = RobustScaler() if cont_scaler == 'r' else StandardScaler()
        # categorical_transformer = OrdinalEncoder()
        # numeric_transformer = Pipeline(
        #     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", cont_scaler)]
        # )
        # self.preprocessor = ColumnTransformer(
        #     transformers=[
        #         ("num", numeric_transformer, numeric_features),
        #         ("cat", categorical_transformer, categorical_features),
        #     ]
        # )
        # self.preprocessor.fit(corpus_whole_view)
    
    def corpus_wise_norm_transform(self, df: pd.DataFrame):
        c = self.cont_preprocessor.transform(df[self.column_cont])
        d = self.disc_preprocessor.transform(df[self.column_disc])
        return np.concatenate([d, c], axis=1)
        # return self.preprocessor.transform(df)
    
    def corpus_wise_norm_rev_transform(self, arr: np.array):
        d = self.disc_preprocessor.inverse_transform(arr[:, :len(self.column_disc)])
        c = self.cont_preprocessor.inverse_transform(arr[:, len(self.column_disc):])
        return np.concatenate([d, c], axis=1)
        # return self.preprocessor.rev_transform(df)

    def transform(self, df: pd.DataFrame):
        data = df.dropna().copy()[self.raw_columns]
        # time stamp and time duration
        data = data.sort_values(by=['te'], ignore_index=True)
        data['te'] = pd.to_datetime(data['te'])
        data['te_stamp'] = data['te'].apply(lambda x: int(round(x.timestamp())))
        data['delta_te'] = data['te_stamp'] - data['te_stamp'].shift(1)
        data = data.fillna(0)

        data['delta_te'] = data['delta_te'].round(0).astype(int)
        data['td'] = data['td'].round(0).astype(int)
        
        self.starting_time_stamp = data['te_stamp'][0]
        data = data.drop(['te', 'te_stamp'], axis=1)
        # log for byt and pkt
        data['byt'] = np.log(data['byt'])
        # data['pkt'] = np.log(data['pkt'])
        # dummy IP and port
        data[['sa_0','sa_1', 'sa_2', 'sa_3']] = data['sa'].str.split('.',expand=True)
        data[['da_0','da_1', 'da_2', 'da_3']] = data['da'].str.split('.',expand=True)
        data = data.drop(['sa', 'da'], axis=1)
        # dummy flg
        data[['flg_0', 'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5']] = data['flg'].apply(lambda x: pd.Series(flg_dump(x)))
        data = data.drop(['flg'], axis=1)

        # transform for individual_based behavior modeling, [sa, da] decide -> [sp, dp]
        if self.individual_user:
            data = swap_attr_at(data,
                    condition=(data["sa_0"].astype(int) != 42) | (data["sa_1"].astype(int) != 219),
                    src_attr_list=['sa_0','sa_1', 'sa_2', 'sa_3', 'sp'],
                    tgt_attr_list=['da_0','da_1', 'da_2', 'da_3', 'dp'])

        # put 'delta_te' the first column
        # first_col = data.pop('delta_te')
        # data.insert(0, 'delta_te', first_col)

        # reorder the column for the convienience
        self.column_disc = ['delta_te', 'lable', 'pr', 'td',
                            'sa_0', 'sa_1' , 'sa_2', 'sa_3', 'da_0', 'da_1', 'da_2', 'da_3',
                            'sp', 'dp', 
                            'flg_0', 'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5',
                            'pkt']
        self.column_cont = ['byt']
        return data[self.column_disc+self.column_cont]

    def rev_transform(self, arr):
        data = pd.DataFrame(arr, columns=self.column_disc+self.column_cont)
        # time stamp
        data['delta_te'] = data['delta_te'].astype(float).round(1)
        data['te_stamp'] = data['delta_te'].cumsum().apply(lambda x: x + self.starting_time_stamp)
        data['te'] = pd.to_datetime(data['te_stamp'], unit='s').dt.round('S')
        data = data.drop(['te_stamp', 'delta_te'], axis=1)
        # log for byt and pkt
        data['td'] = data['td'].astype(float).round(1)
        # try:
        data.loc[data['byt']<1,'byt']= 1
        data.loc[data['byt']>25,'byt']= 25
        data['byt'] = np.exp(data['byt'].astype(float)).round(0).astype(int)
        # dummy IP and port
        if self.individual_user:
            data = swap_attr_at(data,
                    condition=(data["sa_0"].astype(int) != 42) | (data["sa_1"].astype(int) != 219),
                    src_attr_list=['sa_0','sa_1', 'sa_2', 'sa_3','sp'],
                    tgt_attr_list=['da_0','da_1', 'da_2', 'da_3', 'dp'])
            
        data['sa'] = data[['sa_0','sa_1', 'sa_2', 'sa_3']].round(0).astype(int).astype(str).agg('.'.join, axis=1)
        data['da'] = data[['da_0','da_1', 'da_2', 'da_3']].round(0).astype(int).astype(str).agg('.'.join, axis=1)
        data = data.drop(['sa_0','sa_1', 'sa_2', 'sa_3', 'da_0','da_1', 'da_2', 'da_3'], axis=1)
        data['sp'] = data['sp'].astype(float).round(0).astype(int)
        data['dp'] = data['dp'].astype(float).round(0).astype(int)
        # dummy flg
        data['flg'] = data[['flg_0', 'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5']].apply(lambda x: pd.Series(rev_flg_dump(x)), axis=1)
        data = data.drop(['flg_0', 'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5'], axis=1)

        return data[self.raw_columns]

def rectify_output_data(working_dir='tmp'):
    os.makedirs(os.path.join(os.getcwd(), working_dir, 'generated'), exist_ok=True)
    npy_files = glob(os.path.join(os.getcwd(), working_dir, 'nn_output', "*.npy"))

    ntt = NetflowProcessor(working_dir=working_dir).load_processor()
    for i, senario in enumerate(npy_files):
        gen_data = np.load(os.path.join(os.getcwd(), working_dir, 'nn_output', senario)) # load
        gen_data = ntt.corpus_wise_norm_rev_transform(gen_data)
        gen_data_real = ntt.rev_transform(gen_data)
        
        gen_data_real.to_csv(os.path.join(os.getcwd(), working_dir, 'generated', str(i)+'.csv'), index = None)


def rectify_input_data(bcl : BehaviorContextLoader, cont_scaler='r', individual_user=False,
                    in_dir='data_slot', working_dir='tmp'):
    '''
        cont_scaler = 'm' for minmaxscaler or 's' for standardscaler
    '''
    os.makedirs(working_dir+'/transed', exist_ok=True)
    os.makedirs(working_dir+'/normed', exist_ok=True)
    os.makedirs(working_dir+'/nn_input', exist_ok=True)
    path = os.getcwd()
    csv_files = glob(os.path.join(path, in_dir, "*.csv"))  
    
    ntt = NetflowProcessor(individual_user=individual_user, working_dir=working_dir)
    # netflow csv folder -> transfered csv folder
    corpus, fid = [], []
    for f in csv_files:
        df = pd.read_csv(f)
        transed = ntt.transform(df)
        transed.to_csv(os.path.join(path, working_dir, 'transed', f.split('/')[-1]), index=None)
        corpus.append(transed)
        fid.append(f.split('/')[-1])
    print('[Done] netflow transfer')

    ntt.corpus_wise_norm_fit(corpus, cont_scaler=cont_scaler)
    ntt.save_processor()

    # transfered csv folder -> normalized npy folder
    scenario = []
    for data, f in zip(corpus, fid):
        ori_data = ntt.corpus_wise_norm_transform(data)
        np.save(os.path.join(path, working_dir, 'normed', f), ori_data) # save
        # new_num_arr = np.load(os.path.join(path, 'tmp', 'normed', f+'.npy')) # load
        scenario.append(ori_data)
    print('[Done] corpus-wise norm')

    # normalized npy folder -> model data
    temp_event, temp_behavior, temp_next, temp_snap = [], [], [], []
    for ori_data in scenario:
        _short, _next = bcl.short_context_loading(ori_data)
        print('event data scale:', len(_short), _short[0].shape)
        temp_event += _short
        temp_next += _next
    
        _long, _snap = bcl.long_context_loading(ori_data, time_delta_col=0, return_snapshot=True)
        print('behavior data scale:', len(_long), _long[0].shape)
        temp_behavior += _long
        temp_snap += _snap
    print('[Done] load short and long context data')
    snap_df = pd.DataFrame(temp_snap)
    snap_df.to_csv(os.path.join(os.getcwd(), 'results', 'stats', 'snap.csv'), index=False, header=False)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_event))
    data_short, data_long, data_next = [], [], []
    for i in range(len(temp_event)):
        data_short.append(temp_event[idx[i]])
        data_long.append(temp_behavior[idx[i]])
        data_next.append(temp_next[idx[i]])
    print('[Done] mix the datasets (to make it similar to i.i.d)')

    # save data
    np.save(os.path.join(path, working_dir, 'nn_input', 'train_event'), data_short)
    np.save(os.path.join(path, working_dir, 'nn_input', 'train_behavior'), data_long)
    np.save(os.path.join(path, working_dir, 'nn_input', 'train_next'), data_next)
    # new_num_arr = np.load(os.path.join(path, 'tmp', 'nn_input', 'train_event.npy')) # load

    # save data column description
    data_description = [ntt.column_cont, ntt.column_disc, ntt.column_disc_num, bcl.short_seq_len, bcl.long_seq_len]
    print('continuous columns: ', ntt.column_cont)
    print('discrete columns  : ', ntt.column_disc)
    print('discrete columns #: ', ntt.column_disc_num)
    
    with open(os.path.join(path, working_dir, 'nn_input', 'train_desc.pickle'), 'wb') as file:
        pickle.dump(data_description, file)
    # load with
    # with open(os.path.join(path, 'tmp', 'nn_input', 'train_desc.pickle'), 'rb') as filehandle:
    #     data_description = pickle.load(filehandle)

    return data_short, data_long, data_next, data_description
