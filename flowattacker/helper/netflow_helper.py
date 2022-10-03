import numpy as np

_lable_list = [
    'background',       # 0
    'blacklist',        # 1
    'anomaly-udpscan',  # 2
    'dos',              # 3
    'scan11',           # 4
    'scan44',           # 5
    'nerisbotnet',      # 6
]

def lable_dump(lable_str):
    rt = [0] * len(_lable_list)
    rt[_lable_list.index(lable_str)] = 1
    return rt

def rev_lable_dump(lable_dummy):
    rt = _lable_list[np.where(lable_dummy.values == 1)[0][0]]
    return rt
        
def flg_dump(flg_str):
    rt = []
    for x in flg_str:
        rt.append(0 if x == '.' else 1)
    return rt

def rev_flg_dump(flg_dummy):
    flgs = 'UAPRSF'
    rt = ''
    for i in range(6):
        rt += '.' if flg_dummy['flg_%d' % i] == 0 else flgs[i]
    return rt

def pr_dump(pr_str):
    pr_dict = {
        'TCP': 0,
        'UDP': 1,
    }
    rt = [0, 0, 0]
    try:
        rt[pr_dict[pr_str]] = 1
    except:
        rt[-1] = 1
    return rt

def rev_pr_dump(pr_dummy):
    if pr_dummy['pr_0'] == 1:
        return 'TCP'
    elif pr_dummy['pr_1'] == 1:
        return 'UDP'
    else:
        return 'Other'

def swap_attr_at(df, condition, src_attr_list, tgt_attr_list):
    df.loc[condition, src_attr_list+tgt_attr_list] = \
        (df.loc[condition, tgt_attr_list+src_attr_list].values)
    return df