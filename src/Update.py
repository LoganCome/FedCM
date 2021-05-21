import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import torch.nn.functional as F


class DatasetSplitLM(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdateLM(object):
    def __init__(self, args, dataset, idxs, nround, user):
        self.args = args
        self.round = nround
        self.user = user
        self.loss_func = nn.CrossEntropyLoss()
        self.data_loader = DataLoader(DatasetSplitLM(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        list_loss = []
        true_y, pred_y = [], []
        for iter in range(self.args.local_ep):
            for batch_ind, data in enumerate(self.data_loader):
                net.zero_grad()
                # sents.sort(key=lambda l: len(l), reverse=True)
                # print(data)
                x, y = data
                # print(F.one_hot(y, num_classes=self.args.num_classes))
                # print(F.one_hot(y, num_classes=self.args.num_classes).shape)
                # true_y.extend(F.one_hot(y, num_classes=self.args.num_classes))
                true_y.extend(y)
                # print(x.shape, y.shape)
                if self.args.gpu != -1:
                    net = net.cuda()
                    x, y = x.cuda(), y.cuda()
                out = net(x)
                pred_y.extend(torch.argmax(out.cpu(), dim=1))
                loss = self.loss_func(out, y.data)
                loss.backward()
                optimizer.step()
                # Calculate accuracy.
                list_loss.append(loss.item())
        # print(true_y, pred_y)
        # print(len(true_y), len(pred_y))
        # print(len(true_y[0]), len(pred_y[0]))
        # true_y = [t.detach().numpy() for t in true_y]
        # pred_y = [t.detach().numpy() for t in pred_y]
        # print(classification_report(true_y, pred_y, digits=5))
        return {'params': net.cpu().state_dict(),
                'loss': sum(list_loss) / len(list_loss),
                'recall': classification_report(true_y, pred_y, digits=5)
                }

    def evaluate(self, data_loader, model):
        """ Perplexity of the given data with the given model. """
        model.eval()
        with torch.no_grad():
            list_loss = []
            true_y, pred_y = [], []
            for val_idx, data in enumerate(data_loader):
                x, y = data
                true_y.extend(y)
                if self.args.gpu != -1:
                    x, y = x.cuda(), y.cuda()
                    model = model.cuda()
                out = model(x)
                pred_y.extend(torch.argmax(out.cpu(), dim=1))
                loss = self.loss_func(out, y.data)
                # Calculate accuracy.
                list_loss.append(loss.item())
        # print(true_y, pred_y)
        print(classification_report(true_y, pred_y, digits=5))
        return {'loss': sum(list_loss) / len(list_loss),
                'recall': classification_report(true_y, pred_y, digits=5)
                }
