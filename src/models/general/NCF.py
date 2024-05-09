# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from models.general.BPR import BPR


class NCF(BPR):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--layers', type=str, default='[64, 64, 64, 64]',
                            help="Size of each layer.")
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        return BPR.parse_model_args(parser)

    def __init__(self, args, corpus, data):
        self.layers = eval(args.layers)
        self.dropout = args.dropout
        super().__init__(args, corpus, data)

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size, bias=False))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

        self.u_bias = nn.Embedding(self.user_num, 1)
        self.i_bias = nn.Embedding(self.item_num, 1)


    def forward(self, u_ids, i_ids, flag):
        #u_ids = feed_dict['user_id']  # [batch_size]
        #i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)

        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)

        # user_bias = self.u_bias(u_ids).view_as(prediction)
        # item_bias = self.i_bias(i_ids).view_as(prediction)

        # prediction = prediction + user_bias + item_bias
        return prediction.view(len(u_ids), -1)
    

    # inference
    def infer_user_scores(self, u_ids, i_ids):
        # u_ids dimension: (u_size)
        # i_ids dimension: (i_size)

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_u_vectors = mf_u_vectors.unsqueeze(1).repeat(1, i_ids.shape[0], 1)
        mf_i_vectors = mf_i_vectors.unsqueeze(0).repeat(u_ids.shape[0], 1, 1)
        mf_vector = mf_u_vectors * mf_i_vectors

        # print('mf_u_vectors shape: {}'.format(mf_u_vectors.shape))
        # print('mf_i_vectors shape: {}'.format(mf_i_vectors.shape))
        # print('mf_vector shape: {}'.format(mf_u_vectors.shape))
        # print(mf_u_vectors)
        # print(mf_i_vectors)
        # print(mf_vector)

        mlp_u_vectors = mlp_u_vectors.unsqueeze(1).repeat(1, i_ids.shape[0], 1).view(-1, i_ids.shape[0], mlp_u_vectors.shape[-1])
        mlp_i_vectors = mlp_i_vectors.unsqueeze(0).repeat(u_ids.shape[0], 1, 1).view(u_ids.shape[0], -1, mlp_i_vectors.shape[-1])

        # print('mlp_u_vectors shpae {}'.format(mlp_u_vectors.shape))
        # print('mlp_i_vectors shape {}'.format(mlp_i_vectors.shape))
        # print(mlp_u_vectors)
        # print(mlp_i_vectors)

        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        # print('mf_vector shape {}'.format(mf_vector.shape))
        # print('mlp_vector shape {}'.format(mlp_vector.shape))
        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        # print('prediction shape {}'.format(prediction.shape))

        # prediction = prediction.view(u_ids.shape[0], i_ids.shape[0])
        # print('u_ids shape {}'.format(u_ids.shape))
        # print('i_ids shape {}'.format(i_ids.shape))

        scores = prediction.view(u_ids.shape[0], i_ids.shape[0])

        # user_bias = self.u_bias(u_ids).view_as(prediction)
        # item_bias = self.i_bias(i_ids).view_as(prediction)
        # prediction = prediction + user_bias + item_bias


        return scores