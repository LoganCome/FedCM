import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np


def log_x(x):
    _sgn = torch.sign(x)
    _x = torch.abs(x)
    return torch.mul(torch.log(_x + 1), _sgn)


def aggregate_att(w_clients, w_server, stepsize, metric, dp, gamma, user, con):
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters

    :param gamma: forgetting coefficient
    :param user: agents
    :param con: contributions
    """
    w_next = copy.deepcopy(w_server)

    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_server.keys():
        for i in range(0, len(w_clients)):
            _w_server = w_server[k].reshape(-1)
            _w_client = w_clients[i][k].reshape(-1)
            # print(_w_server.shape, _w_client.shape)
            att[k][i] = torch.from_numpy(np.array(np.linalg.norm(_w_server - _w_client, ord=metric)))
    for k in w_server.keys():
        att[k] = F.softmax(att[k], dim=0)

    for k_idx, k in enumerate(w_server.keys()):
        att_weight = torch.zeros_like(w_server[k].reshape(-1), dtype=torch.float32)
        w_c = [[] for i in range(len(w_clients))]
        for i in range(0, len(w_clients)):
            _w_server = w_server[k].reshape(-1)
            _w_client = w_clients[i][k].reshape(-1)
            # print(_w_server.shape, _w_client.shape)
            att_weight += torch.mul(_w_server - _w_client, att[k][i])
            w_c[i] = (stepsize * att[k][i] * (_w_client - _w_server)).clone().detach()
            w_c[i] = torch.true_divide(w_c[i], len(w_clients))

        for i in range(0, len(w_clients)):
            # print(log_x(torch.tensor(w_c[i])))
            # print(log_x(torch.mul(att_weight, stepsize)))
            con[user[i]][k_idx] = con[user[i]][k_idx] * gamma + \
                                  log_x(w_c[i]) - log_x(torch.mul(att_weight, stepsize))

    return w_next
