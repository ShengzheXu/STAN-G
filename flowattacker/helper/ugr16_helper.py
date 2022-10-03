import os
import pandas as pd
import gc
from datetime import datetime

class UGRData(object):
    '''
    Output: download raw data from ugr website https://nesg.ugr.es/nesg-ugr16/
    '''
    def __init__(self) -> None:
        self.source = [("july", "week5"), ("august", "week1"), ("august", "week2")]
        self.big_datafile = os.path.join(os.getcwd(), 'data', 'august.week1.csv')

    def download_background(self, month, week):
        fstr = '%s_%s_csv.tar.gz' % (month, week)
        os.system('wget -P ./data/ https://nesg.ugr.es/nesg-ugr16/download/attack/%s/%s/%s_%s_csv.tar.gz' % (month, week, month, week))
        os.system('tar -C ./data/ -xf ./data/%s' % fstr)

    def download_attack(self, month, week):
        files = ["blacklist_%s_csv.tar.gz", "dos_%s_csv.tar.gz", "scan44_%s_csv.tar.gz", "sshscan_%s_csv.tar.gz",
            "botnet_%s_csv.tar.gz", "scan11_%s_csv.tar.gz", "spam_%s_csv.tar.gz", "udpscan_%s_csv.tar.gz"]

        for f in files:
            fstr = f % '_'.join([month, week])
            os.system('wget -P ./data/ https://nesg.ugr.es/nesg-ugr16/download/attack/%s/%s/%s' % (month, week, fstr))
            os.system('tar -C ./data/ -xf ./data/%s ' % fstr)

    def count(self):
        for month, week in self.source:
            os.system("find ./data/%s/%s -name '*.csv' | xargs wc -l" % (month, week))

    def download(self):
        for month, week in self.source:
            dir = os.path.join(os.getcwd(), 'data', '%s/%s' % (month, week))
            print(dir, os.path.isdir(dir))

            if not os.path.isdir(dir):
                print('downloading:', month, week)
                self.download_attack(month, week)
                self.download_background(month, week)

        self.count()
    
    def sample_ip(self, date='2016-08-01'):
        attacker_ip = ['42.219.150.246', '42.219.150.247', '42.219.150.243', '42.219.150.242', '42.219.150.241'] # A1-A5
        victom_ip = ['42.219.156.30', '42.219.156.31', '42.219.156.29', '42.219.156.28', '42.219.156.27',      # V11-V15
                     '42.219.158.16', '42.219.158.17', '42.219.158.18', '42.219.158.19', '42.219.158.21',      # V21-V25
                     '42.219.152.20', '42.219.152.21', '42.219.152.22', '42.219.152.23', '42.219.152.18',      # V31-V45
                     '42.219.154.69', '42.219.154.68', '42.219.154.70', '42.219.154.71', '42.219.154.66',      # V41-V45                   
                    ]
        def get_normal_ip(normal_ip_str):
            normal_ip = normal_ip_str.split(',')
            return [item for item in normal_ip if (item not in attacker_ip) and (item not in victom_ip)]

        normal_ip_100 = '42.219.156.193,42.219.145.225,42.219.157.220,42.219.159.209,42.219.154.26,42.219.157.9,42.219.152.255,42.219.158.255,42.219.153.109,42.219.154.99,42.219.154.204,42.219.153.129,42.219.152.150,42.219.156.21,42.219.153.135,42.219.153.50,42.219.152.227,42.219.159.249,42.219.153.96,42.219.152.252,42.219.152.238,42.219.157.30,42.219.157.5,42.219.158.153,42.219.158.248,42.219.159.166,42.219.155.144,42.219.158.224,42.219.152.137,42.219.154.53,42.219.155.27,42.219.152.203,42.219.154.60,42.219.159.213,42.219.158.77,42.219.158.214,42.219.158.78,42.219.154.227,42.219.152.142,42.219.154.84,42.219.158.192,42.219.152.189,42.219.152.21,42.219.154.25,42.219.157.27,42.219.154.57,42.219.156.244,42.219.158.232,42.219.152.18,42.219.153.86,42.219.158.194,42.219.158.181,42.219.159.201,42.219.153.202,42.219.158.70,42.219.158.254,42.219.158.197,42.219.158.189,42.219.158.142,42.219.155.185,42.219.153.84,42.219.156.228,42.219.157.14,42.219.153.83,42.219.158.121,42.219.154.61,42.219.158.138,42.219.153.31,42.219.154.47,42.219.145.125,42.219.152.221,42.219.153.52,42.219.158.236,42.219.155.82,42.219.153.172,42.219.154.102,42.219.157.62,42.219.153.16,42.219.153.91,42.219.154.83,42.219.154.253,42.219.158.212,42.219.156.182,42.219.152.29,42.219.158.163,42.219.152.153,42.219.152.92,42.219.155.85,42.219.154.49,42.219.153.56,42.219.159.240,42.219.158.231,42.219.154.12,42.219.158.159,42.219.158.237,42.219.156.241,42.219.158.233,42.219.158.18,42.219.155.177,42.219.158.56'
        normal_ip_100 = get_normal_ip(normal_ip_100)
        normal_ip_10 = '42.219.153.56,42.219.154.49,42.219.154.57,42.219.154.253,42.219.154.60,42.219.156.241,42.219.153.135,42.219.158.142,42.219.153.16,42.219.158.237'
        normal_ip_10 = get_normal_ip(normal_ip_10)
        
        self._extract(self.big_datafile, attacker_ip, date=date, folder_tag='attack_ip')
        self._extract(self.big_datafile, normal_ip_10, date=date, folder_tag='normal_ip_10')
        self._extract(self.big_datafile, normal_ip_100, date=date, folder_tag='normal_ip_100')
        

    def _extract(self, big_datafile, target_ip, date='2016-08-01', chunksize = 10 ** 6, write_to_file=True, folder_tag='attack_ip'):
        os.makedirs(os.path.join(os.getcwd(), 'data', date, folder_tag), exist_ok=True)

        chunkNum = 0
        pass_time = False
        columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
        for chunk in pd.read_csv(big_datafile, chunksize=chunksize, header=None, names = columnName):    
            block_time1 = datetime.now()

            if date is not None:
                chunk = chunk[chunk['te'].str.startswith(date)]

            if len(chunk.index) == 0:
                if pass_time: 
                    break
                else:
                    continue
            pass_time = True
            chunkNum += 1
            
            # chunk = chunk.sample(n=int(len(chunk.index)/10),random_state=131,axis=0)
            for one_ip in target_ip:
                if 'attack' in folder_tag:
                    ip_chunk = chunk[(chunk['sa'] == one_ip)]
                else:
                    ip_chunk = chunk[(chunk['sa'] == one_ip) | (chunk['da'] == one_ip)]
                print('chunk for', one_ip, len(ip_chunk.index))
                if write_to_file:
                    print(len(ip_chunk.index), "to write for", one_ip)
                    
                    outputfile = os.path.join(os.getcwd(), 'data', date, folder_tag, one_ip+'.csv')
                    if not os.path.isfile(outputfile):
                        ip_chunk.to_csv(outputfile, header='column_names', index=False)
                    else: # else it exists so append without writing the header
                        ip_chunk.to_csv(outputfile, mode='a', header=False, index=False)
                del ip_chunk

            block_time2 = datetime.now()
            print("blockNum", chunkNum, ",time:", (block_time2-block_time1).seconds)
            del chunk
            gc.collect()

class ScenarioSampler(object):
    def __init__(self) -> None:
        pass

