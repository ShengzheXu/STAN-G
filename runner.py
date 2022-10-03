import pandas as pd
from flowattacker.model.netflow_processor import NetflowProcessor, rectify_input_data, rectify_output_data
from flowattacker.model.context_loader import BehaviorContextLoader, ScenarioDataset
from flowattacker.model import NetflowAttacker
import torch
import numpy as np
import os
from glob import glob
import distutils.dir_util


def test_netflow_processor():
    df = pd.read_csv('data/demo.csv')
    df = df[df['td']>0][:100]
    print(df)
    print('='*10)

    ntt = NetflowProcessor()
    transed = ntt.transform(df)
    print(transed)
    print('='*10)

    rev = ntt.rev_transform(transed)
    print(rev)

def test_scenario_to_dataslot(merge=True, folder='./data_slot'):
    os.makedirs(folder, exist_ok=True)
    corpus = []
    normal_total = []
    def select_scenario(csv_files, data_type='normal', max_user=100):
        for idx, f in enumerate(csv_files):
            if idx == max_user:
                break
            df = pd.read_csv(f)
            if data_type == 'normal':
                # mask = (df['te'] >= '2016-08-01 00:00:00') & (df['te'] <= '2016-08-01 23:59:59')
                # scenario_df = df.loc[mask]
                # print(len(scenario_df.index))
                mask = (df['te'] >= '2016-08-01 00:00:00') & (df['te'] <= '2016-08-01 23:59:59') & (df['lable'] == 'background') 
                scenario_df = df.loc[mask]
                normal_total.append(len(scenario_df.index))
            elif data_type == 'attack':
                mask0 = (df['te'] >= '2016-08-01 08:00:00') & (df['te'] < '2016-08-01 08:01:00')
                df.loc[mask0,'lable'] = 'dos11'
                # mask1 = (df['te'] >= '2016-08-01 08:10:00') & (df['te'] < '2016-08-01 08:11:00')
                # df.loc[mask1,'lable'] = 'dos53s'
                mask2 = (df['te'] >= '2016-08-01 08:40:00') & (df['te'] < '2016-08-01 08:41:00')
                scenario_df = df.loc[mask0|mask2]
                scenario_df.loc[mask2,'te'] = (pd.to_datetime(scenario_df.loc[mask2,'te']) - pd.Timedelta(seconds=60*39)).dt.strftime('%Y-%m-%d %H:%M:%S')
                print(f.split('/')[-1], len(scenario_df.index))
            elif data_type == 'bot':
                mask = (df['te'] >= '2016-08-01 09:00:00') & (df['te'] < '2016-08-01 09:10:00')
                scenario_df = df.loc[mask].sort_values(by=['te'], ignore_index=True)
                print(f.split('/')[-1], len(scenario_df.index))
            else:
                print('wrong type')

            if merge is False:
                scenario_df.to_csv(os.path.join(os.getcwd(), folder, f.split('/')[-1]), index=None)
            else:
                corpus.append(scenario_df)
    
    csv_files = glob(os.path.join(os.getcwd(), 'data/2016-08-01/normal_ip_100', "*.csv"))
    select_scenario(csv_files, 'normal', max_user=30)
    print('normal total', sum(normal_total))

    csv_files = glob(os.path.join(os.getcwd(), 'data/2016-08-01/attack_ip', "*246.csv"))
    select_scenario(csv_files, 'attack')
    
    csv_files = ['/home/shengzx/proj2/EpisodeTransformer/data/august/week1/botnet_flows_cut.csv']
    select_scenario(csv_files, 'bot')

    if merge:
        df = pd.concat(corpus)
        df = df.sort_values(by=['te'], ignore_index=True)
        df.to_csv(os.path.join(os.getcwd(), folder, 'scenario0.csv'), index=None)

def test_pre(in_dir, working_dir):
    short_seq_len=5
    long_time_delta=60
    long_seq_len=5
    
    bcl = BehaviorContextLoader(short_seq_len=short_seq_len,
                            long_time_delta=long_time_delta,
                            long_seq_len=long_seq_len)

    rectify_input_data(bcl, individual_user=False, in_dir=in_dir, working_dir=working_dir)


def test_model(working_dir='tmp', encoder_type='attn_long', decoder_type='seq', train=False):
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = ScenarioDataset(working_dir)
    print(dataset.description())

    b2i = NetflowAttacker(dataset.data_description, device, encoder_type=encoder_type, decoder_type=decoder_type, save_to=working_dir)

    if train:
        b2i.fit(dataset, num_epochs=6)

    short_seq_len=5
    long_time_delta=60*1
    long_seq_len=5
    bcl = BehaviorContextLoader(short_seq_len=short_seq_len,
                            long_time_delta=long_time_delta,
                            long_seq_len=long_seq_len)

    synth_data = b2i.sample(n=100000, num_scenario = 1, bcl=bcl)

    os.makedirs(os.path.join(os.getcwd(), working_dir , 'nn_output'), exist_ok=True)
    for i, senario in enumerate(synth_data):
        np.save(os.path.join(os.getcwd(), working_dir, 'nn_output', f'train_synthetic_{i}'), senario)
        print(f'wrote synthetic_{i} to nn_output')

def test_post(working_dir):
    rectify_output_data(working_dir=working_dir)


if __name__ == "__main__":
    '''
    passed functions
    '''
    # test_netflow_processor()

    std_dir = 'tmp'
    # test_scenario_to_dataslot(merge=True)
    # test_pre(in_dir='data_slot', working_dir=std_dir)

    working_dir = 'tmp_1min_seq'
    # distutils.dir_util.copy_tree(std_dir, working_dir)
    
    # test_model(working_dir=working_dir, encoder_type='cnn', decoder_type='seq', train=True)
    # test_model(working_dir=working_dir, encoder_type='attn_long', decoder_type='seq', train=True)
    test_post(working_dir=working_dir)
