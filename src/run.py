import random
import torch
import torchvision
from torch import optim, nn
from torch.utils.data import DataLoader

from Update import LocalUpdateLM
from agg.avg import *
from agg.aggregate import *
from Datasets import DatasetLM
from utils.sampling import partition
from utils.options import args_parser
from utils.calcons import cal_con
from Models import RnnLm, Net


def get_batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


if __name__ == "__main__":
    args = args_parser()

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_train = DatasetLM(args.dataset, 'train')
    # dataset_val = DatasetLM(args.dataset, 'valid')
    dataset_test = DatasetLM(args.dataset, 'test')

    loader_train = DataLoader(dataset=dataset_train, batch_size=args.bs, shuffle=False)
    # loader_val = DataLoader(dataset=dataset_val, batch_size=args.bs, shuffle=False)
    loader_test = DataLoader(dataset=dataset_test, batch_size=args.bs, shuffle=False)

    dict_users = partition(len_dataset=len(dataset_train), num_users=args.nusers,
                           dataset=dataset_train, opt=args.opt)

    config = args
    config.num_classes = dataset_train.num_classes
    config.channel = dataset_train.channel
    # print(config.channel)

    if 'resnet18' in str(args.model).lower():
        net_glob = Net(torchvision.models.resnet18(num_classes=config.num_classes), config.channel)
    elif 'resnet101' in str(args.model).lower():
        net_glob = Net(torchvision.models.resnet101(num_classes=config.num_classes), config.channel)
    elif 'resnet34' in str(args.model).lower():
        net_glob = Net(torchvision.models.resnet34(num_classes=config.num_classes), config.channel)
    elif 'resnet50' in str(args.model).lower():
        net_glob = Net(torchvision.models.resnet50(num_classes=config.num_classes), config.channel)
    elif 'vgg16' in str(args.model).lower():
        net_glob = Net(torchvision.models.vgg16(num_classes=config.num_classes), config.channel)
    elif 'resnext50' in str(args.model).lower():
        net_glob = Net(torchvision.models.resnext50_32x4d(num_classes=config.num_classes), config.channel)
    else:
        print('No such model')
        exit(1)

    if args.gpu != -1:
        net_glob = net_glob.cuda()
    w_glob = net_glob.cpu().state_dict()

    con = [[] for i in range(args.nusers)]
    for i in range(args.nusers):
        con[i] = [[] for j in range(len(w_glob.keys()))]
        for k_idx, k in enumerate(w_glob.keys()):
            con[i][k_idx] = torch.zeros_like(w_glob[k].reshape(-1))

    lr = args.lr
    best_val_loss = None
    model_saved = './log/model_{}_{}_{}_{}_{}_{}.pt'.format(args.model, args.opt, args.dataset, args.epochs, args.agg,
                                                            args.frac)
    loss_saved = './log/loss_{}_{}_{}_{}_{}_{}.npy'.format(args.model, args.opt, args.dataset, args.epochs, args.agg,
                                                           args.frac)
    con_saved = './log/con_{}_{}_{}_{}_{}_{}.npy'.format(args.model, args.opt, args.dataset, args.epochs, args.agg,
                                                         args.frac)
    result_saved = './log/result_{}_{}_{}_{}_{}_{}.npy'.format(args.model, args.opt, args.dataset, args.epochs,
                                                               args.agg, args.frac)
    contemp_saved = './log/contemp_{}_{}_{}_{}_{}_{}.npy'.format(args.model, args.opt, args.dataset, args.epochs,
                                                                 args.agg, args.frac)

    loss_train = []
    result_train = []

    try:
        for epoch in range(args.epochs):
            net_glob.train()
            w_locals, loss_locals, recall_locals = [], [], []
            m = max(int(args.frac * args.nusers), 1)
            idxs_users = np.random.choice(range(args.nusers), m, replace=False)
            result_epoch = []
            for idx in idxs_users:
                print('-' * 30, f'User {idx}', '-' * 30)
                local = LocalUpdateLM(args=args, dataset=dataset_train, idxs=dict_users[idx], nround=epoch, user=idx)
                net_glob.load_state_dict(w_glob)
                out_dict = local.update_weights(net=copy.deepcopy(net_glob))
                w_locals.append(copy.deepcopy(out_dict['params']))
                loss_locals.append(copy.deepcopy(out_dict['loss']))
                recall_locals.append(copy.deepcopy(out_dict['recall']))
                result_epoch.append([idx, out_dict['recall']])
            result_train.append(result_epoch)

            # update global weights
            if args.agg == 'avg':
                w_glob = average_weights(w_locals, w_glob, args.epsilon, args.ord, dp=args.dp,
                                         gamma=args.gamma, user=idxs_users, con=con)
            elif args.agg == 'att':
                print(args.gamma, idxs_users)
                w_glob = aggregate_att(w_locals, w_glob, args.epsilon, args.ord, dp=args.dp,
                                       gamma=args.gamma, user=idxs_users, con=con)
            else:
                exit('Unrecognized aggregation')
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            if args.epochs % 10 == 0:
                print('\nTrain loss:', loss_avg)
            loss_train.append(loss_avg)

            # save temp con
            with open(contemp_saved, 'a') as f:
                print(cal_con(con))
                print(cal_con(con), file=f)

            print('-' * 30, f'Val', '-' * 30)
            val_result = local.evaluate(data_loader=loader_train, model=net_glob)
            print("Epoch {}".format(epoch))
            # print("Validation acc:", "\n", val_result)

            if not best_val_loss or val_result['loss'] < best_val_loss:
                with open(model_saved, 'wb') as f:
                    torch.save(net_glob, f)
                best_val_loss = val_result['loss']
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
        print(loss_train)
        print(val_result['loss'])
        final_loss = loss_avg
        # div_loss = []
        # recall_loss = []

        # -------------------------------------------- #
        '''
        for i in range(config.nusers):
            print('-' * 40, i, '-' * 40)
            net_glob = RnnLm(config)
            if args.gpu != -1:
                net_glob = net_glob.cuda()
            w_glob = net_glob.cpu().state_dict()
            best_val_loss = None
            loss_train = []

            set_s = set(range(args.nusers))
            set_s = set_s - {i}
            for epoch in range(args.epochs):
                net_glob.train()
                w_locals, loss_locals, recall_locals = [], [], []
                m = max(int(args.frac * args.nusers), 1)
                idxs_users = np.random.choice(range(args.nusers), m, replace=False)
                for idx in idxs_users:
                    local = LocalUpdateLM(args=args, dataset=dataset_train, idxs=dict_users[idx], nround=epoch,
                                          user=idx)
                    net_glob.load_state_dict(w_glob)
                    out_dict = local.update_weights(net=copy.deepcopy(net_glob))
                    w_locals.append(copy.deepcopy(out_dict['params']))
                    loss_locals.append(copy.deepcopy(out_dict['loss']))
                    recall_locals.append(copy.deepcopy(out_dict['recall']))

                # update global weights
                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)

                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                if args.epochs % 10 == 0:
                    print('\nTrain loss:', loss_avg)
                loss_train.append(loss_avg)

                val_result = local.evaluate(data_loader=loader_val, model=net_glob)
                print("Epoch {}, Validation ppl: {:.1f}".format(epoch, val_result))
            print(loss_train)
            div_loss.append(loss_avg)
        '''

    except KeyboardInterrupt:
        print('-' * 89)
        print('Existing from training early')

    for i in range(len(con)):
        for j in range(len(con[i])):
            con[i][j] = np.mean(con[i][j].detach().numpy())
        con[i] = np.mean(con[i])
        print(con[i])

    for i in range(args.nusers):
        print(len(dict_users[i]), end=', ')
    print('')

    # save train result
    print(loss_train, file=open(loss_saved.replace('npy', 'txt'), 'w'))
    print(con, file=open(con_saved.replace('npy', 'txt'), 'w'))
    print(result_train, file=open(result_saved.replace('npy', 'txt'), 'w'))

    import pickle

    pickle.dump(loss_train, file=open(loss_saved.replace('npy', 'pkl')))
    pickle.dump(con, file=open(con_saved.replace('npy', 'pkl')))
    pickle.dump(result_train, file=open(result_saved.replace('npy', 'pkl')))

    np.save(loss_saved, np.array(loss_train))
    np.save(con_saved, np.array([t.detach().numpy() for t in con]))
    np.save(result_saved, np.array(result_train))

    # Load the best saved model.
    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    pp_train = local.evaluate(data_loader=loader_train, model=model_best)
    # pp_val = local.evaluate(data_loader=loader_val, model=model_best)
    pp_test = local.evaluate(data_loader=loader_test, model=model_best)

    print("Train perplexity: {}".format(pp_train))
    # print("Val perplexity: {:.1f}".format(pp_val))
    print("Test perplexity: {}".format(pp_test))

    print('final_loss:', final_loss)
    # print('div_loss:', div_loss)
    # print('ppl_loss:', ppl_loss)
