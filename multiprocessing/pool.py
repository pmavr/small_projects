import multiprocessing
from itertools import product
from time import time


def merge_names(a, b):
    return '{} & {}'.format(a, b)


if __name__ == '__main__':
    start = time()
    names = [['Brown', 1], ['Wilson', 2], ['Bartlett', 3], ['Rivera', 4], ['Molloy', 5], ['Opie', 6]]
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(merge_names, names)
    end = time() - start
    print(f'Elapsed time: {end}\n{results}')


