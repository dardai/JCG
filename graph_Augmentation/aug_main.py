import numpy as nm
import scipy.sparse as sp
import datetime
import graph_Augmentation.randomAugment as ra
from logging import getLogger
from recbole.utils import set_color


def inter_random_augment(original_interactions, density_sample_ratio):

    num_of_user = original_interactions.shape[0]
    num_of_item = original_interactions.shape[1]

    logger = getLogger()
    logger.info(set_color('non-zero element in original_interactions', 'green')
                + f': {original_interactions.count_nonzero()}')

    density = original_interactions.nnz / (num_of_user * num_of_item)
    logger.info(set_color('density', 'yellow') + f': {density}')
    logger.info(set_color('density_sample_ratio', 'green') + f': {density_sample_ratio}')

    if density > 0.01:
        random_augment_inter_matrix = ra.sample_random_matrix_augment(original_interactions,
                                                                      density_sample_ratio).tocoo()
    else:
        random_augment_inter_matrix = ra.large_random_matrix_augment(original_interactions,
                                                                     density_sample_ratio).tocoo()
    logger.info(set_color('non-zero element in random_augment', 'green')
                + f': {random_augment_inter_matrix.count_nonzero()}')

    logger.info(f'{datetime.datetime.now()}' + set_color(': calculate the enhancement matrix', 'green'))

    return random_augment_inter_matrix


def aug_filter(sparse_matrix, gate):
    aug_tuple = sp.find(sparse_matrix >= gate)
    user_index = aug_tuple[0]
    item_index = aug_tuple[1]
    ratings = nm.ones_like(user_index, dtype=nm.float32)
    shape = sparse_matrix.shape
    aug_matrix = sp.coo_matrix((ratings, (user_index, item_index)), shape=shape)
    return aug_matrix
