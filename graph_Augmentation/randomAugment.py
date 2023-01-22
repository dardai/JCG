import random
import numpy as nm
import scipy.sparse as sp
from datetime import datetime
from logging import getLogger
from recbole.utils import set_color


def random_graph_augment(liked_graph, sample_proportion=0.1, augment_value=1):
    augmented_liked_graph = liked_graph.copy()
    zero_index = nm.argwhere(augmented_liked_graph == 0)
    num_of_zero = len(zero_index)
    sample_num = int(num_of_zero * sample_proportion)
    random_index = random.sample(zero_index.tolist(), sample_num)
    for index in random_index:
        augmented_liked_graph[index[0], index[1]] = augment_value

    return augmented_liked_graph


def find_random_matrix_augment(sparse_matrix, sample_proportion=0.1, augment_value=1):
    augment_matrix = sp.lil_matrix(sparse_matrix)
    zero_tuple = sp.find(augment_matrix == 0)
    zero_index = list(zip(zero_tuple[0], zero_tuple[1]))
    num_of_zero = len(zero_index)
    sample_num = int(num_of_zero * sample_proportion)
    random_index = random.sample(zero_index, sample_num)
    for index in random_index:
        augment_matrix[index[0], index[1]] = augment_value
    return augment_matrix


def sample_random_matrix_augment(sparse_matrix, sample_proportion=0.1, augment_value=1):
    logger = getLogger()
    logger.info(f'{datetime.now()}' + set_color(': sample_random_matrix_augment begin', 'green'))

    augment_matrix = sp.lil_matrix(sparse_matrix)
    sample_num = int(sparse_matrix.nnz * sample_proportion)
    random_index = sample_zero_n(sparse_matrix, sample_num)
    for index in random_index:
        augment_matrix[index[0], index[1]] = augment_value
    return augment_matrix


def large_random_matrix_augment(sparse_matrix, sample_proportion=0.1, augment_value=1):
    logger = getLogger()
    logger.info(f'{datetime.now()}' + set_color(': large_random_matrix_augment begin', 'green'))
    augment_matrix = sp.lil_matrix(sparse_matrix)

    sample_num = int(sparse_matrix.nnz * sample_proportion)
    for _ in range(sample_num):
        sampled_index = nm.random.randint(0, [sparse_matrix.shape[0], sparse_matrix.shape[1]])
        augment_matrix[sampled_index[0], sampled_index[1]] = augment_value

    return augment_matrix


def sample_zero_n(mat, n):
    itr = sample_zero_forever(mat)
    return [next(itr) for _ in range(n)]


def sample_zero_forever(mat):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = tuple(nm.random.randint(0, [mat.shape[0], mat.shape[1]]))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)
