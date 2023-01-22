# -*- coding: utf-8 -*-
r"""
JCG
"""

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from graph_Augmentation.aug_main import inter_random_augment


class JCG(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(JCG, self).__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_user_e = None
        self.restore_item_e = None

        self.leave_ratio = 0.2
        self.density_sample_ratio = config['density_sample_ratio']
        # print("density_sample_ratio", self.density_sample_ratio)
        self.aug_inter_matrix1 = self.inter_matrix_add_augment()
        self.aug_inter_matrix2 = self.inter_matrix_add_augment()
        print("aug_matrix1", self.aug_inter_matrix1.shape)
        print("aug_matrix2", self.aug_inter_matrix2.shape)

        self.norm_adj_mat = self.get_norm_adj_mat(self.interaction_matrix).to(self.device)
        self.aug_norm_mat1 = self.get_norm_adj_mat(self.aug_inter_matrix1).to(self.device)
        self.aug_norm_mat2 = self.get_norm_adj_mat(self.aug_inter_matrix2).to(self.device)

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.user_conv_centroids = None
        self.user_conv_2cluster = None
        self.item_conv_centroids = None
        self.item_conv_2cluster = None

        self.aug_user_conv_centroids1 = None
        self.aug_user_conv_2cluster1 = None
        self.aug_item_conv_centroids1 = None
        self.aug_item_conv_2cluster1 = None

        self.aug_user_conv_centroids2 = None
        self.aug_user_conv_2cluster2 = None
        self.aug_item_conv_centroids2 = None
        self.aug_item_conv_2cluster2 = None

    def update_HYP_kmeans(self):
        aug_user_all_emb1, aug_item_all_emb1, _ = self.forward(self.aug_norm_mat1)
        aug_user_conv_embeddings1 = aug_user_all_emb1.detach().cpu().numpy()
        aug_item_conv_embeddings1 = aug_item_all_emb1.detach().cpu().numpy()
        self.aug_user_conv_centroids1, self.aug_user_conv_2cluster1 = self.run_kmeans(aug_user_conv_embeddings1)
        self.aug_item_conv_centroids1, self.aug_item_conv_2cluster1 = self.run_kmeans(aug_item_conv_embeddings1)

        aug_user_all_emb2, aug_item_all_emb2, _ = self.forward(self.aug_norm_mat2)
        aug_user_conv_embeddings2 = aug_user_all_emb2.detach().cpu().numpy()
        aug_item_conv_embeddings2 = aug_item_all_emb2.detach().cpu().numpy()
        self.aug_user_conv_centroids2, self.aug_user_conv_2cluster2 = self.run_kmeans(aug_user_conv_embeddings2)
        self.aug_item_conv_centroids2, self.aug_item_conv_2cluster2 = self.run_kmeans(aug_item_conv_embeddings2)

    def update_COP_kmeans(self):
        user_conv_embeddings, item_conv_embeddings, _ = self.forward(self.norm_adj_mat)

        user_conv_embeddings = user_conv_embeddings.detach().cpu().numpy()
        item_conv_embeddings = item_conv_embeddings.detach().cpu().numpy()

        self.user_conv_centroids, self.user_conv_2cluster = self.run_kmeans(user_conv_embeddings)
        self.item_conv_centroids, self.item_conv_2cluster = self.run_kmeans(item_conv_embeddings)

    def update_INP_kmeans(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()

        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=False)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)
        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self, inter_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = inter_matrix
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        index = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(index, data, torch.Size(L.shape))
        return SparseL

    def inter_matrix_add_augment(self):
        aug_int_mat = inter_random_augment(self.interaction_matrix, self.density_sample_ratio)
        return aug_int_mat

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, norm_adj_mat):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = torch.sparse.mm(norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def AugConvNCE_loss(self, aug_user_conv_embeddings_all, aug_item_conv_embeddings_all, user, item, aug_order):
        if aug_order == 1:
            aug_user_conv_2cluster = self.aug_user_conv_2cluster1
            aug_user_conv_centroids = self.aug_user_conv_centroids1
            aug_item_conv_2cluster = self.aug_item_conv_2cluster1
            aug_item_conv_centroids = self.aug_item_conv_centroids1
        # elif aug_order == 2:
        else:
            aug_user_conv_2cluster = self.aug_user_conv_2cluster2
            aug_user_conv_centroids = self.aug_user_conv_centroids2
            aug_item_conv_2cluster = self.aug_item_conv_2cluster2
            aug_item_conv_centroids = self.aug_item_conv_centroids2
        # assert aug_order in range(1, 2)
        # aug_user_conv_embeddings_all, aug_item_conv_embeddings_all = torch.split(node_embedding,
        #                                                                         [self.n_users, self.n_items])

        aug_user_conv_embeddings = aug_user_conv_embeddings_all[user]
        norm_aug_user_conv_embeddings = F.normalize(aug_user_conv_embeddings)

        user2cluster = aug_user_conv_2cluster[user]
        user2centroids = aug_user_conv_centroids[user2cluster]
        pos_score_user = torch.mul(norm_aug_user_conv_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_aug_user_conv_embeddings, aug_user_conv_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        aug_conv_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        aug_item_conv_embeddings = aug_item_conv_embeddings_all[item]
        norm_aug_conv_item_embeddings = F.normalize(aug_item_conv_embeddings)

        item2cluster = aug_item_conv_2cluster[item]
        item2centroids = aug_item_conv_centroids[item2cluster]
        pos_score_item = torch.mul(norm_aug_conv_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_aug_conv_item_embeddings, aug_item_conv_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        aug_conv_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        conv_nce_loss = self.proto_reg * (aug_conv_nce_loss_user + aug_conv_nce_loss_item)
        return conv_nce_loss

    def ConvNCE_loss(self, node_embedding, user, item):
        user_conv_embeddings_all, item_conv_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_conv_embeddings = user_conv_embeddings_all[user]
        norm_user_conv_embeddings = F.normalize(user_conv_embeddings)
        user2cluster = self.user_conv_2cluster[user]
        user2centroids = self.user_conv_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_conv_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_conv_embeddings, self.user_conv_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        conv_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_conv_embeddings = item_conv_embeddings_all[item]
        norm_conv_item_embeddings = F.normalize(item_conv_embeddings)

        item2cluster = self.item_conv_2cluster[item]
        item2centroids = self.item_conv_centroids[item2cluster]
        pos_score_item = torch.mul(norm_conv_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_conv_item_embeddings, self.item_conv_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        conv_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        conv_nce_loss = self.proto_reg * (conv_nce_loss_user + conv_nce_loss_item)
        return conv_nce_loss

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]
        norm_user_embeddings = F.normalize(user_embeddings)
        user2cluster = self.user_2cluster[user]
        user2centroids = self.user_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]
        item2centroids = self.item_centroids[item2cluster]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)

        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def ssl_graph_loss(self, aug_user_all_emb1, aug_item_all_emb1,
                       aug_user_all_emb2, aug_item_all_emb2, user, item):

        norm_all_user_emb = F.normalize(aug_user_all_emb2)
        aug_user_all_emb1 = aug_user_all_emb1[user]
        aug_user_all_emb2 = aug_user_all_emb2[user]
        norm_user_emb1 = F.normalize(aug_user_all_emb1)
        norm_user_emb2 = F.normalize(aug_user_all_emb2)

        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        aug_item_all_emb1 = aug_item_all_emb1[item]
        previous_item_embeddings = aug_item_all_emb2[item]
        norm_item_emb1 = F.normalize(aug_item_all_emb1)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(aug_item_all_emb2)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def aug_ssl_layer_loss(self, aug_user_all_emb, aug_item_all_emb, previous_embedding, user, item):
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

        aug_user_all_emb = aug_user_all_emb[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(aug_user_all_emb)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        aug_item_all_emb = aug_item_all_emb[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(aug_item_all_emb)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward(self.norm_adj_mat)

        center_embedding = embeddings_list[0]

        context_embedding = embeddings_list[self.hyper_layers * 2]

        aug_user_all_emb1, aug_item_all_emb1, _ = self.forward(self.aug_norm_mat1)
        aug_user_all_emb2, aug_item_all_emb2, _ = self.forward(self.aug_norm_mat2)

        # INP_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)
        COP_loss = self.ConvNCE_loss(center_embedding, user, pos_item)
        # HYP_loss = self.AugConvNCE_loss(aug_user_all_emb1, aug_item_all_emb1, user, pos_item, 1)+
        #              self.AugConvNCE_loss(aug_user_all_emb2, aug_item_all_emb2, user, pos_item, 2)

        # HES_loss = self.aug_ssl_layer_loss(aug_user_all_emb1, aug_item_all_emb1, center_embedding, user, pos_item)+
        #            self.aug_ssl_layer_loss(aug_user_all_emb2, aug_item_all_emb2, center_embedding, user, pos_item)
        HOS_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        # HYP_loss = self.ssl_graph_loss(aug_user_all_emb1, aug_item_all_emb1, aug_user_all_emb2, aug_item_all_emb2, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, HOS_loss, COP_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward(self.norm_adj_mat)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward(self.norm_adj_mat)
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
