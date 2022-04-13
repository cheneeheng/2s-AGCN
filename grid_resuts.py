import os
import numpy as np
import matplotlib.pyplot as plt


def grid_cmulti():
    GRID_DIR = 'data/data/ntu_result/xview/sgn_v6/grid_cmulti'
    LOG_PATH = [os.path.join(GRID_DIR, i, 'log.txt')
                for i in sorted(os.listdir(GRID_DIR))
                if '.' not in i]

    for path in LOG_PATH:
        if not os.path.exists(path):
            print(path, 'does not exists...')

    result_f = open(os.path.join(GRID_DIR, 'results.txt'), 'w+')

    acc = []
    for path_i, path in enumerate(LOG_PATH):
        with open(path, 'r') as log_f:
            for line_i, line in enumerate(log_f):
                if line_i == 963:
                    vals_str = path.split('/')[-2][19:]
                    list_vals = []
                    x, y, z = True, True, True
                    for v in vals_str:
                        if int(v) == 1:
                            list_vals.append('1.00')
                            x, y, z = True, True, True
                        elif int(v) == 2:
                            y = False
                            list_vals.append('0.25')
                        elif int(v) == 5:
                            if y:
                                list_vals.append('0.50')
                            x, y, z = True, True, True
                    print(list_vals, line[-7:-1], file=result_f)
                    acc.append(float(line[-7:-2]))

    result_f.close()

    acc = np.array(acc).reshape((-1, 9))
    plt.imshow(acc, cmap='jet')
    plt.xticks([i for i in range(9)],
               ['C3 0.25\nC4 0.25',
                '0.25\n0.50',
                '0.25\n1.00',
                '0.50\n0.25',
                '0.50\n0.50',
                '0.50\n1.00',
                '1.00\n0.25',
                '1.00\n0.50',
                '1.00\n1.00'])
    plt.yticks([i for i in range(9)],
               ['C1 0.25\nC2 0.25',
                'C1 0.25\nC2 0.50',
                'C1 0.25\nC2 1.00',
                'C1 0.50\nC2 0.25',
                'C1 0.50\nC2 0.50',
                'C1 0.50\nC2 1.00',
                'C1 1.00\nC2 0.25',
                'C1 1.00\nC2 0.50',
                'C1 1.00\nC2 1.00'])
    plt.savefig(os.path.join(GRID_DIR, 'results.png'))
    # plt.show()


def grid_seg():
    GRID_DIR = 'data/data/ntu_result/xview/sgn_v6/grid_seg'
    LOG_PATH = [os.path.join(GRID_DIR, i, 'log.txt')
                for i in sorted(os.listdir(GRID_DIR))
                if '.' not in i]

    for path in LOG_PATH:
        if not os.path.exists(path):
            print(path, 'does not exists...')

    result_f = open(os.path.join(GRID_DIR, 'results.txt'), 'w+')

    var = []
    acc = []
    for path_i, path in enumerate(LOG_PATH):
        with open(path, 'r') as log_f:
            for line_i, line in enumerate(log_f):
                if line_i == 963:
                    var += [path.split('/')[-2][-2:]]
                    print(var[-1], line[-7:-1], file=result_f)
                    acc.append(float(line[-7:-2]))

    result_f.close()

    acc = np.array(acc).reshape((1, -1))
    plt.imshow(acc, cmap='jet')
    plt.xticks([i for i in range(len(var))], var)
    plt.savefig(os.path.join(GRID_DIR, 'results.png'))
    # plt.show()


if __name__ == '__main__':
    # grid_cmulti()
    grid_seg()
