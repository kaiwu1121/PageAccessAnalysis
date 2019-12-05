import os
import numpy as np
import random
import csv
import torch
import torchvision
from io import StringIO
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# data preparation
def get_address(root, trace_file, split=False):
    """
    :param root: file folder
    :param trace_file: the original trace file name
    :return: address_train.txt contains the 70% addresses, address_val.txt contains the rest 30% addresses
    """
    with open(os.path.join(root, trace_file)) as f:
        if split:
            with open('{}/address_train.txt'.format(root), 'w') as t:
                with open('{}/address_val.txt'.format(root), 'w') as v:
                    content = f.readlines()
                    len_train = int(round(0.7 * len(content)))
                    print('len of the trace: {}, len of the train: {}'.format(len(content), len_train))

                    for i in range(len(content)):
                        address = content[i].split(',')[0].split()[1]
                        address = int(address, 16)

                        if i < len_train:
                            t.write('{}\n'.format(address))
                        else:
                            v.write('{}\n'.format(address))
        else:
            with open('{}/address_all.txt'.format(root), 'w') as a:
                content = f.readlines()
                for i in range(len(content)):
                    address = content[i].split(',')[0].split()[1]
                    address = int(address, 16)

                    a.write('{}\n'.format(address))


# get_address(root='data/1204/d_24', trace_file='cg.d.24t.log')


def plot_2darr(arr1, arr2, name):
    plt.plot(arr1, arr2, 'bo')
    plt.title("delta and target")
    plt.xlabel("delta")
    plt.ylabel("taget of delta")
    plt.savefig(name)
    plt.close()


def plot_arr(arr, name):
    plt.plot(arr[0:2000])
    # plt.title(name)
    plt.xlabel("Index in dataset")
    plt.ylabel("{}".format(name))
    plt.show()
    plt.close()


class DataPrepare:
    # cluster the raw addresses, then compute the delta within the cluster
    def __init__(self, root):
        self.root = root                    # folder
        self.raw_add_train = 'address_train.txt'
        self.raw_add_val = 'address_val.txt'       # address_train.txt
        self.app_addr = 'app_addr.txt'

        self.train_root = os.path.join(self.root, 'train')
        self.val_root = os.path.join(self.root, 'val')

        self.n_cluster = 6                  # number of clusters
        self.page_size = 1                 # divide the addresses by this number
        self.frequency = 1                 # we will consider the delta if it appears more than 10 times
        self.target_delta_len = None

    def cluster(self):
        '''
        :return: cluster the raw addresses, store the results to dict_cluster, and write each cluster to one txt file
                 '{root}/{key}cluster_address.txt
        '''
        # write all the address to dict with cluster
        dict_cluster = self.initiDict(self.n_cluster)

        # raw address -> reshape -> cluster -> store in dict {cluster label: [addresses within the cluster]}
        # !!!! change the addr !!!!
        raw_address = self.load_address_array(self.app_addr)
        reshape_address = np.reshape(raw_address, (-1, 1))

        self.kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(reshape_address)
        center = self.kmeans.cluster_centers_
        clusters_labels = KMeans(n_clusters=self.n_cluster, random_state=0).fit_predict(reshape_address)

        for i in range(len(reshape_address)):
            dict_cluster[str(clusters_labels[i])].append(raw_address[i])

        # write one cluster to one txt file !!!! change the folder !!!!
        self.dict2txt(dict_cluster, self.root)

        # predict the label of val set !!!! uncomment this for val !!!!
        # self.cluster_val()
        return dict_cluster, center

    def cluster_val(self):
        dict_cluster = self.initiDict(self.n_cluster)
        raw_address = self.load_address_array(self.raw_add_val)

        u, c = np.unique(raw_address, return_counts=True)

        reshape_address = np.reshape(raw_address, (-1, 1))

        predict_labels = self.kmeans.predict(reshape_address)

        for i in range(len(reshape_address)):
            dict_cluster[str(predict_labels[i])].append(raw_address[i])

        # write one cluster to one txt file
        self.dict2txt(dict_cluster, self.val_root)
        return dict_cluster

    def get_delta(self, adderss_file, delta_file, folder):
        # compute the delta given addresses, then store in the txt file
        with open('{}'.format(adderss_file), 'r') as a:
            address = a.readlines()
            address = [int(int(i)/self.page_size) for i in address]

            address_f = address[:-1]
            address_b = address[1:]
            delta = np.subtract(address_b, address_f)

            with open('{}/{}'.format(folder, delta_file), 'w') as d:
                for item in delta:
                    d.write("%s\n" % item)
        return delta

    def initiDict(self, n_cluster):
        # initialize the dict to empty list for each key
        dict_cluster = {}
        for i in range(n_cluster):
            dict_cluster[str(i)] = []
        return dict_cluster

    def load_address_array(self, address_txt):
        # load the addresses to array
        addressfile = open('{}/{}'.format(self.root, address_txt), 'r')
        all_address = np.asarray(addressfile.readlines(), dtype=int)
        all_address = all_address.astype(int)
        addressfile.close()
        return all_address

    def dict2txt(self, dict_cluster, folder):
        # save the dict to n_cluster txt files
        for key in sorted(dict_cluster):
            file = open('{}/{}_cluster_address.txt'.format(folder, key), 'w')
            for address in dict_cluster[key]:
                file.write('{}\n'.format(address))
            file.close()

    def delta_hist(self, folder):
        for key in range(self.n_cluster):
            dict_delta = {}     # {unique delta: (#. occur of the delta, #. target deltas, {target: #. occur}}
            cluster_address = os.path.join(folder, '{}_cluster_address.txt'.format(key))
            delta_in_cluster = self.get_delta(cluster_address, '{}_cluster_delta.txt'.format(key), folder)

            for i in range(len(delta_in_cluster) - 1):
                # if it is not a unique delta
                if str(delta_in_cluster[i]) in dict_delta.keys():
                    target = dict_delta[str(delta_in_cluster[i])][2]

                    if str(delta_in_cluster[i + 1]) not in target.keys():
                        target[str(delta_in_cluster[i + 1])] = 1
                    else:
                        target[str(delta_in_cluster[i + 1])] = target[str(delta_in_cluster[i + 1])] + 1

                    dict_delta[str(delta_in_cluster[i])] = (dict_delta[str(delta_in_cluster[i])][0] + 1, len(target), target)

                else:
                    dict_delta[str(delta_in_cluster[i])] = (1, 1, {str(delta_in_cluster[i + 1]): 1})

            # get the target delta
            delta_key_list = [int(z) for z in dict_delta.keys()]
            # write the delta dict to key_delta_his.txt
            his_delta = open('{}/{}.txt'.format(folder, str(key) + '_unique_delta'), 'w')
            for key_delta in sorted(delta_key_list):
                his_delta.write('{},{}\n'.format(key_delta, dict_delta[str(key_delta)]))

            dict_common_next = self.get_common_delta(dict_delta, key, delta_key_list, folder)
            self.remove_noise_target(dict_common_next, key, folder)

    def get_common_delta(self, dict_unique_delta, cluster, delta_key_list, folder):
        f = open('{}/{}_common_delta.txt'.format(folder, cluster), 'w')
        dict_common_delta = {}
        for key in sorted(delta_key_list):
            freq = dict_unique_delta[str(key)][0]
            if freq >= self.frequency:
                dict_common_delta[str(key)] = dict_unique_delta[str(key)]
                f.write('{},{}\n'.format(key, dict_unique_delta[str(key)]))

        print('cluster label is {}\t n is:{} '.format(cluster, len(dict_common_delta)))
        return dict_common_delta

    def remove_noise_target(self, dict_c_delta, cluster, folder):
        c_key_list = [int(z) for z in dict_c_delta.keys()]
        with open('{}/{}.txt'.format(folder, str(cluster) + '_delta_target'), 'w') as f:
            for key in sorted(c_key_list):
                target_dict = dict_c_delta[str(key)][2]
                max_target_key = max(target_dict, key=target_dict.get)
                max_target_times = target_dict[max_target_key]
                f.write('{},{}\n'.format(key, max_target_key))


def main():
    trace = DataPrepare(root='data/1204/500_1')
    # 1. cluster the address file to clusters, the addresses of each cluster in '{}_cluster_address.txt'
    address_cluster_dict, center_value = trace.cluster()
    # 2. compute the delta with in each cluster, in '{}_cluster_delta.txt'
    # 3. get the unique deltas and their next delta with in each cluster, in '{}_unique_delta.txt'
    #    delta: #. occurrence, #. unique next delta, #. next delta dict(next delta: #. occur of the next delta)
    # 4. get the common deltas, #. occurrence > 10, {}_common_delta.txt
    # 5. get the common delta, and the target from the next delta dict, get the most frequent next delta as target
    #    in '{}_delta_target.txt', each line: delta, target
    trace.delta_hist(trace.root)
    # trace.delta_hist(trace.val_root)

#
# if __name__ == '__main__':
#     main()


def make_dataset(file_list):
    samples = None
    for file in file_list:
        f = open(str(file), 'r')
        f_array = np.loadtxt(f, delimiter=',')

        if samples is None:
            samples = f_array
        else:
            samples = np.vstack((samples, f_array))
    return samples


class SeqDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.n_cluster = 6
        self.sequence_len = 1
        txt_paths = sorted(Path(self.root).glob('*_{}'.format('delta_target.txt')))
        self.file_lists = [z for z in txt_paths if z.stem[0] is not '.']
        if len(self.file_lists) != self.n_cluster:
            raise(RuntimeError('Found {} txt files, but {} clusters'.format(len(self.file_lists), self.n_cluster)))

        self.samples = make_dataset(self.file_lists)
        target_classes, indices, counts = self._find_classes()
        self.classes = target_classes
        self.class_to_idx = indices

        self.deltas = self.samples[:, 0]
        self.targets = self.samples[:, 1]

        self.min_delta = -4297426.0
        self.max_delta = 4668141.0

        # plot_arr(self.deltas, 'deltas')
        # plot_arr(self.targets, 'targets')

    def _plot_cluster(self):
        color_list = ['red', 'orange', 'green', 'blue', 'magenta', 'black']
        fig, ax = plt.subplots()

        for clu in range(self.n_cluster):
            txt_paths = sorted(Path(self.root).glob('{}_{}'.format(str(clu), 'delta_target.txt')))
            self.file_lists = [z for z in txt_paths if z.stem[0] is not '.']

            samples = make_dataset(self.file_lists)
            deltas = samples[:, 0]
            targets = samples[:, 1]

            color = color_list[clu]
            ax.scatter(deltas, targets, c=color, label=color, alpha=0.3, edgecolors='none')

        # plot_arr(self.deltas, '{}deltas'.format(str(clu)))
        # plot_arr(self.targets, '{}targets'.format(str(clu)))
        plt.title("delta and target")
        plt.xlabel("delta")
        plt.ylabel("taget of delta")
        ax.legend()
        plt.show()

    def _find_classes(self):
        target = self.samples[:, 1]
        # target_classes: unique class(next delta)
        # indices: class to index for each target
        # counts: #. occurrence for each unique class
        target_classes, indices, counts = np.unique(target, return_inverse=True, return_counts=True)
        return target_classes, indices, counts

    def __getitem__(self, item):
        seq = self.deltas[item: item+self.sequence_len]
        target = self.targets[item: item+self.sequence_len]

        # labels = self.class_to_idx[item: item+self.sequence_len]
        # labels = labels[-1]
        return seq, target - self.min_delta     # make sure the min target is >= 0

    def __len__(self):
        return len(self.samples) - self.sequence_len





