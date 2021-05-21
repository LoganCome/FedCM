# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/7 18:14
# Description:

import numpy as np


def cal_con(con):
    _con = []
    for i in range(len(con)):
        _con_i = []
        for j in range(len(con[i])):
            _con_i.append(np.mean(con[i][j].detach().numpy()))
        _con.append(np.mean(np.array(_con_i)))
    return _con
