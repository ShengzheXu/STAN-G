from flowattacker.helper.eval_helper import PlotHelper, DomainKnowledgeHelper, MetricHelper, PrivacyHelper
from flowattacker.model import NetflowAttacker, NetflowProcessor
from flowattacker.model.context_loader import ScenarioDataset
import numpy as np
import pandas as pd
import os
import torch
from glob import glob

def test_tsne():
    eval = PlotHelper()
    # eval.test_color()

    data_behavior = np.load(os.path.join(os.getcwd(), 'tmp', 'nn_input', 'train_behavior.npy')) # load
    data_next = np.load(os.path.join(os.getcwd(), 'tmp', 'nn_input', 'train_next.npy')) # load
    ntt = NetflowProcessor().load_processor()

    data_color_idx, data_group = ntt.get_col_by_name('lable')
    data_color = data_next[:, data_color_idx].astype(int)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = ScenarioDataset()
    b2i = NetflowAttacker(dataset.data_description, device)
    data_behavior_emb = b2i.encode_long(dataset)
    eval.plot_tsne(data_behavior_emb, data_color, data_group)


def eval_pipline_single(csv_files):
    eval_metric = MetricHelper()
    corpus_df = []
    for f in csv_files:
        df = pd.read_csv(f)
        corpus_df.append(df)
    
    data_df = pd.concat(corpus_df).sort_values(by=['te'], ignore_index=True)
    print(np.log(data_df['byt'].min()), np.log(data_df['byt'].max()))
    
    eval_metric.count_by_min(data_df)

    eval_dk = DomainKnowledgeHelper()
    eval_dk.check(data_df)
    # print('domain_knowledge score:', domain_score)


def eval_pipline_compare(real_csv_files, synth_csv_files):
    def get_corpus(csv_files):
        corpus_df = []
        for f in csv_files:
            df = pd.read_csv(f)
            corpus_df.append(df)
        return pd.concat(corpus_df).sort_values(by=['te'], ignore_index=True)
    
    real_df = get_corpus(real_csv_files)
    synth_df = get_corpus(synth_csv_files)
    
    eval_privacy = PrivacyHelper()
    eval_privacy.train_exist(real_df, synth_df)

def eval_plots():
    eval_metric = PlotHelper()
    eval_metric.test_color()

if __name__ == "__main__":
    # test_tsne()
    # test_domain_knowledge()
    # eval_plots()
    # train_data
    # train_csv_files = glob(os.path.join(os.getcwd(), 'data', '2016-08-01', 'attack_ip', "*.csv"))
    train_csv_files = glob(os.path.join(os.getcwd(), 'data_slot', "*.csv"))
    eval_pipline_single(train_csv_files)

    # synthetic_data
    # synth_csv_files = glob(os.path.join(os.getcwd(), 'tmp_1min', 'generated', "*.csv"))
    synth_csv_files = glob(os.path.join(os.getcwd(), 'tmp_1min', 'generated', "*.csv"))
    eval_pipline_single(synth_csv_files)
    
    eval_pipline_compare(train_csv_files, synth_csv_files)


    