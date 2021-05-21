import numpy as np
import skimage
import torch
import torchvision
import random


def partition(len_dataset, num_users, dataset, opt='normal'):
    num_items = int(len_dataset / num_users)
    dict_users, all_idxs = {}, [i for i in range(len_dataset)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    if opt == 'less':
        ll = len(dict_users[0]) * 0.2
        for i in range(num_users - 2, num_users):
            while len(dict_users[i]) > ll:
                dict_users[i].pop()
    elif opt == 'less_rank':
        ll = len(dict_users[0]) * 0.50
        for i in range(num_users - 4, num_users - 2):
            while len(dict_users[i]) > ll:
                dict_users[i].pop()
        ll = len(dict_users[0]) * 0.30
        for i in range(num_users - 2, num_users):
            while len(dict_users[i]) > ll:
                dict_users[i].pop()
    elif opt == 'random':
        def F(item):
            x, y = item
            x = torch.randn(x.shape)
            # img = torchvision.transforms.ToPILImage()(x)
            # img.show()
            # img.save(f'./log/random_examples/{y}.bmp')
            item = x, y
            return item
        dataset.setup(F, opt)

        for i in range(num_users - 2, num_users):
            for idx in dict_users[i]:
                dataset.add_item(idx)

    elif opt == 'noise':
        def F(item):
            x, y = item
            x = torch.randn(x.shape)
            # img = torchvision.transforms.ToPILImage()(x)
            # img.show()
            # img.save(f'./log/random_examples/{y}.bmp')
            item = x, y
            return item

        dataset.setup(F, opt)

        for i in range(num_users - 2, num_users):
            for idx in dict_users[i]:
                dataset.add_item(idx)

    elif opt == 'mislabel':
        def F(item):
            x, y = item
            y = random.randint(0, 9)
            # print(y)
            item = x, y
            return item

        dataset.setup(F, opt)

        for i in range(num_users - 2, num_users):
            for idx in dict_users[i]:
                dataset.add_item(idx)
                # x, y = dataset[idx]
                # print(x.shape, y)
    elif opt == 'normal':
        pass
    else:
        print('No such option')
        exit(1)

    return dict_users
