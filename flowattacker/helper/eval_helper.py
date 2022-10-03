import numpy as np
import pandas as pd
import os
import time

from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = "serif"

from .netflow_helper import _lable_list

class PlotHelper(object):
    def __init__(self) -> None:
        self.color_set = ['lightcoral', 'coral', 'darkorange', 'gold', 'palegreen', 'paleturquoise', 'skyblue', 'plum', 'hotpink', 'pink']
        self.color_set2 = ['darkorchid', 'salmon', 'red', 'lightgreen', 'green', 'cyan', 'blue', 'orchid', 'purple', 'grey', 'black']
        pass

    def test_color(self):
        fig, ax = plt.subplots()

        def make_line(color='darkorchid', bias=0):
            stock = lambda A, amp, angle, phase: A * angle + amp * np.sin(angle + phase)
            N = 9
            x = np.linspace(0, 3*np.pi, N)
            mean_stock = (stock(.1, .2, x, 1.2)) + bias
            upper_stock = mean_stock + np.random.randint(N) * 0.02
            lower_stock = mean_stock - np.random.randint(N) * 0.015
        
            # ax.plot(x, mean_stock, color = 'darkorchid', label = r'$y = \gamma \sin(\theta + \phi_0)$')
            ax.plot(x, mean_stock, color = color, label = color)
            ax.fill_between(x, upper_stock, lower_stock, alpha = .1, color = color)
            ax.grid(alpha = .2)
        
        for i, g in enumerate(np.unique(self.color_set)):
            make_line(g, i)
        ax.legend()

        os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'test_color.png'))
    
    def plot_lines(self, result_df):
        fig, ax = plt.subplots()

        def make_line(color='darkorchid', bias=0):
            stock = lambda A, amp, angle, phase: A * angle + amp * np.sin(angle + phase)
            N = 9
            x = np.linspace(0, 3*np.pi, N)
            mean_stock = (stock(.1, .2, x, 1.2)) + bias
            upper_stock = mean_stock + np.random.randint(N) * 0.02
            lower_stock = mean_stock - np.random.randint(N) * 0.015
        
            # ax.plot(x, mean_stock, color = 'darkorchid', label = r'$y = \gamma \sin(\theta + \phi_0)$')
            ax.plot(x, mean_stock, color = color, label = color)
            ax.fill_between(x, upper_stock, lower_stock, alpha = .1, color = color)
            ax.grid(alpha = .2)
        
        for i, g in enumerate(np.unique(self.color_set)):
            make_line(g, i)
        ax.legend()

        os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'test_color.png'))

    def plot_scatters(self, result_df):
        # unique, counts = np.unique(x_color, return_counts=True)
        
        # fig, ax = plt.subplots()
        # for g in unique:
        #     ix = np.where(x_color == g)
        #     x = x_embedded[ix, 0][:, :lim]
        #     y = x_embedded[ix, 1][:, :lim]
        #     ax.scatter(x, y, c = self.color_set2[g], label = x_group[g], s = 2)
        # ax.legend()

        # # plt.xlabel('variables x')
        # # plt.ylabel('variables y')
        # # plt.legend(markerscale=8)

        # os.makedirs(os.path.join(os.getcwd(), 'results', 'tsne'), exist_ok=True)
        # plt.savefig(os.path.join(os.getcwd(), 'results', 'tsne', 'lable.png'))
        pass

    def _compute_tsne(self, data_behavior):
        # make embedding
        # data_behavior = data_behavior.reshape(-1, data_behavior.shape[1] * data_behavior.shape[2])
        data_behavior = data_behavior
        start_time = time.time()
        x_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data_behavior)
        print(f'TSNE Excution Time: {time.time() - start_time:.2f} Seconds')

        return x_embedded #, x_group

    def plot_tsne(self, data_behavior, data_next, data_group):
        unique, counts = np.unique(x_color, return_counts=True)
        print(np.min(counts))
        input()
        lim = 250

        # data_behavior, data_next = data_behavior[:3000], data_next[:3000]
        x_embedded = self._compute_tsne(data_behavior)
        x_color = data_next
        x_group = data_group

        fig, ax = plt.subplots()
        for g in unique:
            ix = np.where(x_color == g)
            x = x_embedded[ix, 0][:, :lim]
            y = x_embedded[ix, 1][:, :lim]
            ax.scatter(x, y, c = self.color_set2[g], label = x_group[g], s = 2)
        ax.legend()

        # plt.xlabel('variables x')
        # plt.ylabel('variables y')
        # plt.legend(markerscale=8)

        os.makedirs(os.path.join(os.getcwd(), 'results', 'tsne'), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'tsne', 'lable.png'))

    def plot_attn(self,):
        pass


class MetricHelper(object):
    def __init__(self) -> None:
        os.makedirs(os.path.join(os.getcwd(), 'results', 'stats'), exist_ok=True)

    def count_by_min(self, data_df, standard='2016-08-01'):
        count = pd.crosstab(data_df['te'], data_df['lable'])        
        count.to_csv('./results/stats/count_by_min.csv')

class PrivacyHelper(object):
    def __init__(self) -> None:
        os.makedirs(os.path.join(os.getcwd(), 'results', 'privacy'), exist_ok=True)

    def train_exist(self, real_df, synth_df):
        out = pd.merge(real_df, synth_df, how='inner')
        out.to_csv('./results/privacy/count_by_min.csv')
        print('duplicated rows:', len(out.index))

class DomainKnowledgeHelper(object):
    '''
    Args:
    - data_df: generated csv data
    '''
    def __init__(self) -> None:
        pass 
        
    def _check_attack(self, data_df):
        # attack percent
        print(data_df['lable'].value_counts())
        # print(data_df['pr'].value_counts())
        # print(data_df['lable'].value_counts(normalize=True))

        # attack 1: victom port is 80 for the scheduled Dos Attack
        df_base = data_df[data_df['lable'] == 'dos11']
        len_base = len(df_base.index)
        len_true = len(df_base[df_base['dp'] == 80].index)
        if len_base == 0:
            print('attack1:', 'Nan')
        else:
            print(f'attack1: {len_true/len_base:.2f}, {len_true}, {len_base} ')

        # attack 2: port scan attacks send SYN flag to victims, the packet is 1 and byte amount is 44
        df_base = data_df[(data_df['lable'] == 'scan11') | (data_df['lable'] == 'scan44')]
        len_base = len(df_base.index)
        len_true = len(df_base[(df_base['pkt'] == 1) & (df_base['byt'] == 44)].index)
        if len_base == 0:
            print('attack2:', 'Nan')
        else:
            print(f'attack2: {len_true/len_base:.2f}, {len_true}, {len_base} ')

    def _check_normal(self, data_df):
        # normal 1: IP address validity
        reserved_ip_set = ['0.0.0.0', '10.0.0.0', '100.64.0.0', '127.0.0.0', '169.254.0.0',
            '172.16.0.0', '192.0.0.0', '192.0.2.0', '192.88.99.0', '192.168.0.0', '198.18.0.0',
            '198.51.100.0', '203.0.113.0', '224.0.0.0', '240.0.0.0', '255.255.255.255']
        len_base = len(data_df.index)
        data_df['norm1'] = data_df.apply(lambda x: 1 if x['sa'] in reserved_ip_set or x['da'] in reserved_ip_set else 0, axis=1)
        # df_true = df_[df_['norm1'] == 0]
        len_true = len(data_df[data_df['norm1'] == 0].index)
        # len_true = len_base - len((data_df['sa'] in reserved_ip_set) | (data_df['da'] in reserved_ip_set))
        print(f'normal1: {len_true/len_base:.2f}, {len_true}, {len_base} ')

        # normal 2: value range of bytes and packets
        df_base_tcp = data_df[data_df['pr']== 'TCP']
        len_base_tcp = len(df_base_tcp.index)
        len_true_tcp = len(df_base_tcp[df_base_tcp['byt']>=40].index)

        df_base_udp = data_df[data_df['pr']== 'UDP']
        len_base_udp = len(df_base_udp.index)
        len_true_udp = len(df_base_udp[df_base_udp['byt']>=28].index)

        len_base = len_base_tcp + len_base_udp
        len_true = len_true_tcp + len_true_udp
        print(f'normal2: {len_true/len_base:.2f}, {len_true}, {len_base} ')

        # normal 3: relationship between bytes and packets
        df_base_tcp = data_df[data_df['pr']== 'TCP']
        len_base_tcp = len(df_base_tcp.index)
        # len_true_tcp = len(df_base_tcp['pkt']*40<=df_base_tcp['byt']<=df_base_tcp['pkt']*65535)
        len_true_tcp = len(df_base_tcp[df_base_tcp['byt'].between(df_base_tcp['pkt']*40, df_base_tcp['pkt']*65535)].index)

        df_base_udp = data_df[data_df['pr']== 'UDP']
        len_base_udp = len(df_base_udp.index)
        # len_true_udp = len(df_base_udp['pkt']*28<=df_base_udp['byt']<=df_base_udp['pkt']*65535)
        len_true_udp = len(df_base_udp[df_base_udp['byt'].between(df_base_udp['pkt']*28, df_base_udp['pkt']*65535)].index)

        len_base = len_base_tcp + len_base_udp
        len_true = len_true_tcp + len_true_udp
        print(f'normal3: {len_true/len_base:.2f}, {len_true}, {len_base} ')

        # normal 4: if pr is not TCP, the flow should not have TCP flags
        df_base = data_df[data_df['pr'] != 'TCP']
        len_base = len(df_base.index)
        len_true = len(df_base[df_base['flg']=='.A....'].index)
        print(f'normal4: {len_true/len_base:.2f}, {len_true}, {len_base} ')

        # normal 5: if user's port number is 80 or 443, the pr must be TCP
        df_base = data_df[(data_df['sp']==80)|(data_df['sp']==443)|(data_df['dp']==80)|(data_df['dp']==443)]
        len_base = len(df_base.index)
        len_true = len(df_base[df_base['pr'] == 'TCP'].index)
        print(f'normal5: {len_true/len_base:.2f}, {len_true}, {len_base} ')

    def check(self, data_df):
        print('full set amount', len(data_df.index))
        self._check_attack(data_df)
        self._check_normal(data_df)

class ApplicationHelper(object):
    '''
    Args:
    - real_train_df: generated csv data

    '''
    def __init__(self, data_real_train, data_real_test, data_synth) -> None:
        self.data_real_train = data_real_train
        self.data_real_test = data_real_test
        self.data_synth = data_synth
    
    def get_data(self, split, x_col, y_col):
        if split == 'train':
            return self.data_real_train[x_col], self.data_real_train[y_col]
        elif split == 'test':
            return self.data_real_test[x_col], self.data_real_test[y_col]
        else: # split == 'synth':
            return self.data_synth[x_col], self.data_synth[y_col]
    
    def tool_test_false_alarm(self,):
        x_col, y_col = [''], ['lable']
        X_train, y_train = self.get_data('train', x_col, y_col)
        X_test, y_test = self.get_data('test', x_col, y_col)
        X_synth, y_synth = self.get_data('synth', x_col, y_col)
        
        clf = MLPClassifier(random_state=1, max_iter=50).fit(X_train, y_train)
        score_real = clf.score(X_test, y_test)
        score_synth = clf.score(X_synth, y_synth)
        print('tool_test_false_alarm:', 'score real', score_real, 'score_synth', score_synth)

    def tool_test_attack_detection(self,):
        x_col, y_col = [''], ['lable']
        X_train, y_train = self.get_data('train', x_col, y_col)
        X_test, y_test = self.get_data('test', x_col, y_col)
        X_synth, y_synth = self.get_data('synth', x_col, y_col)
        
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        score_real = clf.score(X_test, y_test)
        score_synth = clf.score(X_synth, y_synth)
        print('tool_test_false_alarm:', 'score real', score_real, 'score_synth', score_synth)


    def ml_training_forcasting():
        pass

    def ml_training_classification(self):
        pass

    def check(self):
        self.tool_test_false_alarm()
        self.tool_test_attack_detection()
        self.ml_training_classification()
        self.ml_training_forcasting()

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sklearn.svm import SVC

# def f1_validation(y_true, y_pred):
#     # performance
#     print(metrics.classification_report(y_true, y_pred))
#     print(metrics.multilabel_confusion_matrix(y_true, y_pred))
#     #print("F1 micro: %1.4f\n" % f1_score(y_test, y_predicted, average='micro'))
#     print("F1 macro: %1.4f\n" % metrics.f1_score(y_true, y_pred, average='macro'))
#     #print("F1 weighted: %1.4f\n" % f1_score(y_test, y_predicted, average='weighted'))
#     #print("Accuracy: %1.4f" % (accuracy_score(y_test, y_predicted)))
#     return metrics.f1_score(y_true, y_pred, average='macro')

# def balance_vol(df, attack):
#     df_avg1 = df[df['y_lable'] == 0].sample(n=30000)
#     df_avg2 = df[df['y_lable'] == attack]
#     df_avg = df_avg1.append(df_avg2, ignore_index=True)
#     df_avg = df_avg.sample(frac=1)
#     return df_avg

# if __name__ == '__main__':
#     attack = 1
#     df = pd.read_csv('../data/exp3/train/'+'box/flow_all.csv')
#     df = df[(df['y_lable'] == 0) | (df['y_lable'] == attack)]

#     print(df['y_lable'].value_counts())
#     df = balance_vol(df, attack)
#     print(df['y_lable'].value_counts())

#     X = df[['y_ua_0', 'y_ua_1', 'y_ua_2', 'y_ua_3',
#         'y_ta_0', 'y_ta_1', 'y_ta_2', 'y_ta_3', 'y_up_enc', 'y_tp_enc', 'dir', 'pr_0', 'pr_1','pr_2', 'flg_0',
#        'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5', 'log_pkt', 'log_byt']]
#     Y = df['y_lable']
#     y = Y.values

#     df2 = pd.read_csv('../data/exp3/test/'+'box/behav_all.csv')
#     df2 = df2[(df2['y_lable'] == 0) | (df2['y_lable'] == attack)]
#     print(df2['y_lable'].value_counts())
#     # df2 = balance_vol(df2)
#     # print(df2['y_lable'].value_counts())
#     X_test = df2[['y_ua_0', 'y_ua_1', 'y_ua_2', 'y_ua_3',
#         'y_ta_0', 'y_ta_1', 'y_ta_2', 'y_ta_3', 'y_up_enc', 'y_tp_enc', 'dir', 'pr_0', 'pr_1','pr_2', 'flg_0',
#        'flg_1', 'flg_2', 'flg_3', 'flg_4', 'flg_5', 'log_pkt', 'log_byt']]
#     Y_test = df2['y_lable']
#     y_true = Y_test.values

#     # df = pd.read_csv('../data/exp1/test/digit/42.219.145.125.csv')
#     # df2 = pd.read_csv('../data/exp1/test/digit/42.129.145.225.csv')
#     # X = df.drop(['te', 'lable'], axis=1)
#     # y = df['lable'].values
#     # X_test = df2.drop(['te', 'lable'], axis=1)
#     # y_true = df2['lable'].values
    
#     # print(df['lable'].value_counts())

#     print(y)
#     # lp = RandomForestClassifier(n_estimators=100)
#     lp = SVC(kernel='rbf', class_weight='balanced',)
#     lp.fit(X, y)
#     # print(lp.feature_importances_)
#     y_pred = lp.predict(X_test)
#     lr_score = f1_validation(y_true, y_pred)



# 8e_rule_check

# if __name__ == "__main__":
#     df = pd.read_csv('../data/exp3/train/gen/all_gen.csv', header=None)
#     # df = pd.read_csv('../data/exp3/train/raw/scan11.csv', header=None)
#     print(df)

#     df['pkt'] = np.exp(df[19])
#     df['byt'] = np.exp(df[20])
#     # input()




#     


# 9e_traffic_check
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# output_folder = '../data/exp3/train/'

# if __name__ == "__main__":
#     # df = pd.read_csv('../data/exp3/train/gen/all_gen.csv', header=None)
#     df = pd.read_csv('../data/exp3/train/box/flow_all.csv', header=None)
#     print(df.columns)
#     input()
#     # df = df[:20]
#     x = range(len(df.index))
#     y = df[1].tolist()
    
#     plt.plot(x, y, 'o', markersize=1)
#     # plt.legend(markerscale=8)
#     plt.savefig(output_folder+'eval/traffic.png')


# flow plot 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import glob
# plt.style.use('ggplot')
# output_folder = '../data/exp4/train/'

# lable_dict = {
#     'background': 0,
#     'blacklist': 1,
#     'anomaly-udpscan': 2,
#     'dos': 3,
#     'scan11': 4,
#     'scan44': 5,
#     'nerisbotnet': 6,
# }

# colors = cm.rainbow(np.linspace(0, 1, len(lable_dict.keys())))

# def plot_flow(filename):
#     df = pd.read_csv(filename)
#     df['date'] = df['te'].str[8:10].astype(int)
#     df['hour'] = df['te'].str[11:13].astype(int)
#     c = lable_dict[df['lable'][0]]
#     x_list = []
#     x, cnt = [], 0
#     occurence_list = []
#     for day in range(1, 8):
#         for hour in range(0, 24):
#             df_i = df[(df['date'] == day) & (df['hour'] == hour)]
#             x.append(cnt)
#             cnt += 1
#             x_list.append('%d-%d' % (day, hour))
#             # x_list.append(day+0.01*hour)
#             occurence_list.append(len(df_i.index))
#     print(len(occurence_list), occurence_list)
    
#     plt.plot(x, occurence_list, 'o', color=colors[c], markersize=1, label=c)

# if __name__ == "__main__":
#     norm_files = glob.glob('../data/exp4/test/raw/*.csv')
#     for filename in norm_files:
#         plot_flow(filename)
#     # plt.legend(markerscale=1)
#     # plt.legend(markerscale=1), loc='upper right', bbox_to_anchor=(0.5, -0.05))
#     plt.savefig('traffic7.png')