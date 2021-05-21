import copy
import torch
import numpy as np
# from scipy import linalg
import torch.nn.functional as F
import random


def log_x(x):
    _sgn = torch.sign(x)
    _x = torch.abs(x)
    return torch.mul(torch.log(_x+1), _sgn)


def average_weights(w_clients, w_server, stepsize, metric, dp, gamma, user, con):
    """
    Federated averaging
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters

    :param gamma: forgetting coefficient
    :param user: agents
    :param con: contributions
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w_clients[0])
    for k in w_avg.keys():
        for i in range(1, len(w_clients)):
            w_avg[k] = w_avg[k] + w_clients[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w_clients)) + torch.mul(torch.randn(w_avg[k].shape), dp)

    att, att_mat = {}, {}
    for k in w_server.keys():
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
        att_weight += w_avg[k].reshape(-1) - w_server[k].reshape(-1)
        w_c = [[] for i in range(len(w_clients))]
        for i in range(0, len(w_clients)):
            _w_server = w_server[k].reshape(-1)
            _w_client = w_clients[i][k].reshape(-1)
            # print(_w_server.shape, _w_client.shape)
            # att_weight += torch.mul(_w_client - _w_server, att[k][i])
            w_c[i] = (stepsize * att[k][i] * (_w_server - _w_client)).clone().detach()
            w_c[i] = torch.true_divide(w_c[i], len(w_clients))

        for i in range(0, len(w_clients)):
            # print(log_x(torch.tensor(w_c[i])))
            # print(log_x(torch.mul(att_weight, stepsize)))
            con[user[i]][k_idx] = con[user[i]][k_idx] * gamma + \
                                  log_x(w_c[i]) - log_x(torch.mul(att_weight, stepsize))
    return w_avg


