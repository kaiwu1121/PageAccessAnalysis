import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import figure


def load_address_array(root, address_txt):
    # load the addresses to array
    addressfile = open('{}/{}'.format(root, address_txt), 'r')
    all_address = np.asarray(addressfile.readlines(), dtype=int)
    all_address = all_address.astype(int)
    addressfile.close()
    return all_address


def get_delta(adderss_file):
    # compute the delta given addresses, then store in the txt file
    with open('{}'.format(adderss_file), 'r') as a:
        address = a.readlines()
        address = [int(i) for i in address]
        print(len(address))

        address_f = address[:-1]
        address_b = address[1:]
        delta = np.subtract(address_b, address_f)

        # with open('{}/{}'.format(folder, delta_file), 'w') as d:
        #     for item in delta:
        #         d.write("%s\n" % item)
    return delta


def get_overlap():
    folder = 'data/1204'
    file1 = os.path.join(folder, '1')
    file2 = os.path.join(folder, '2')
    file3 = os.path.join(folder, '3')

    name = 'address_all.txt'

    raw_address1 = load_address_array(file1, name)
    raw_address2 = load_address_array(file2, name)
    raw_address3 = load_address_array(file3, name)

    u1, c1 = np.unique(raw_address1, return_counts=True)
    u2, c2 = np.unique(raw_address2, return_counts=True)
    u3, c3 = np.unique(raw_address3, return_counts=True)

    overlap = np.intersect1d(u1, u2)
    overlap = np.intersect1d(overlap, u3)

    return overlap


def remove_overlap(root, file):
    overlap = get_overlap()
    raw_address = load_address_array(root, file)
    print('len of total page accesses: {}'.format(len(raw_address)))

    for o in overlap:
        raw_address = raw_address[raw_address != o]
        # print(len(raw_address))

    with open('{}/{}'.format(root, 'app_addr'), 'w') as d:
        for item in raw_address:
            d.write("%s\n" % item)

    print('len of total application page accesses: {}'.format(len(raw_address)))
    return raw_address


def plot_arr(arr, name, y_ax):
    plt.plot(arr)
    # plt.title(name)
    plt.xlabel("Interval")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name+y_ax)
    plt.close()


def get_interval(subfolder, address_txt, num_chunk):
    raw_address = load_address_array(subfolder, address_txt)
    # plot_arr(raw_address, 'app_addr')

    print('len of original address: {}'.format(len(raw_address)))
    raw_address = raw_address[0:(len(raw_address) // num_chunk * num_chunk)]
    print('len of sliced address: {}'.format(len(raw_address)))

    reshape_address = np.reshape(raw_address, (num_chunk, -1))

    # res_list1 = [[] for x in range(num_chunk - 1)]
    # for i in range(num_chunk - 1):
    #     for j in range(i + 1, num_chunk):
    #         overlap = np.intersect1d(reshape_address[i], reshape_address[j])
    #         res_list1[i].append(len(overlap))

    overlap_res_list2 = []
    unique_res = []
    for m in range(num_chunk - 1):
        j = m+1
        overlap = np.intersect1d(reshape_address[m], reshape_address[j])
        u, c = np.unique(reshape_address[m], return_counts=True)
        unique_res.append(len(u))
        overlap_res_list2.append(len(overlap))

    arr = overlap_res_list2
    # for i in range(num_chunk - 1):
    #     arr.append(res_list1[i][0])

    plot_arr(arr, '1B_24t', 'Duplicated pages')
    plot_arr(unique_res, '1B_24t', 'Num unique addresses')

    # with open('{}/address_overlap_len_{}.txt'.format(subfolder, num_chunk), 'w') as f:
    #     for item in res_list:
    #         f.write("%s\n" % item)

    return overlap_res_list2


def plot_cdf(arr, name, y_ax):
    x = [(x + 1) * 100 for x in range(len(arr))]
    plt.plot(x, arr)
    # plt.title(name)
    plt.xlabel("Top X frequent accessed pages")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name+y_ax)
    plt.close()


def plot_pdf(arr, name, y_ax):
    x = [(x + 1) * 100 for x in range(len(arr))]
    plt.plot(x, arr)
    # plt.title(name)
    plt.xlabel("Page number")
    plt.ylabel("{}".format(y_ax))
    plt.title(name)
    plt.savefig(name+y_ax)
    plt.close()


def main():
    # remove overlap address(system address)
    root = '/home/yuxin/PycharmProjects/address_prediction/data/1204/1_24'
    file = 'address_all.txt'
    app_address = remove_overlap(root, file)

    # get unique application address
    # u, c = np.unique(app_address, return_counts=True)
    # print('len of unique application page accesses: {}'.format(len(u)))

    # offset = max(u) - min(u)
    # print('maximal page access offset: {}'.format(offset))
    # n = len(u)

    # ## CDF
    # sort_c = sorted(c, reverse=True)
    # cdf = []
    # for i in range(0, n // 100):
    #     cdf.append(np.sum(sort_c[0: (i+1)*100]))
    #
    # s = 4018400
    # new_cdf = [x / s for x in cdf]
    # plot_cdf(new_cdf, 'D_24t', 'CDF')
    # print('over')

    ## PDF
    # if not np.array_equal(sorted(u), u):
    #     raise ValueError('u')

    # pdf = []
    # for i in range(0, n//100):
    #     pdf.append(np.sum(c[i*100: (i+1)*100]))

    # plot_pdf(pdf, '1B_1t', 'PDF')

    hist, ed = np.histogram(app_address, bins=21228688)

    plt.plot(hist)
    plt.savefig('05_1_24_pdf.tif')


    # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')




    # print(np.sum(pdf))
    # compute unique delta
    # add_file = '/home/yuxin/PycharmProjects/address_prediction/data/1204/500_1/app_addr.txt'
    # delta = get_delta(add_file)
    # u, c = np.unique(delta, return_counts=True)
    # print('len of unique delta: {}'.format(len(u)))

    # ## get interval
    # root = '/home/yuxin/PycharmProjects/address_prediction/data/1204/1_24'
    # address_txt = 'app_addr.txt'
    # num_chunk = 400
    # res_list = get_interval(root, address_txt, num_chunk)


if __name__ == '__main__':
    main()

















