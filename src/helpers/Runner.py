# -*- coding: UTF-8 -*-

import os
import gc
import copy
import torch
import logging
import numpy as np
import random
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn

from utils import utils
from models.Model import Model

import matplotlib.pyplot as plt
import Inference


class Runner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=500,
                            help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=10,
                            help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='',
                            help='')

        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None  # will store [start_time, last_step_time]

        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch
        self.DRM = args.DRM


    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'adam':
            #logging.info("Optimizer: Adam")
            #if 'parameters' in self.DRM:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
            # else:
            #     optimizer = torch.optim.Adam(
            #     model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    
    def make_plot(self, args, data, name, snap_idx=0):
        y = data
        x = range(len(y))
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(name))
        plt.title('{}_{}'.format(name, snap_idx))
        plt.savefig(args.test_result_file+'_{}_{}.png'.format(name, snap_idx))
        plt.close()

    def train(self,
              model,
              data_dict,
              args,
              corpus,
              snap_idx):
        
        logging.info('Training time stage: {}'.format(snap_idx))

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        # load pretrained model pretrained on historical data
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            #print(model.model_path+'_snap{}'.format(idx-1))
            model.load_model(model.model_path+'_snap{}'.format(snap_idx-1))

        # pretrain model
        if snap_idx > 0 and 'pretrain' in args.dyn_method:
            model.load_model(model.model_path+'_snap0')
            v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
            t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
            logging.info("Pretrained model testing")

            val_str = Inference.print_results(None, v_results, None)
            logging.info(val_str)
            val_result_filename_ = os.path.join(self.test_result_file, 'val_snap{}.txt'.format(snap_idx))
            open(val_result_filename_, 'w+').write(val_str)

            test_str = Inference.print_results(None, None, t_results)
            logging.info(test_str)
            result_filename_ = os.path.join(self.test_result_file, 'test_snap{}.txt'.format(snap_idx))
            open(result_filename_, 'w+').write(test_str)

            return 0, 0


        self._check_time(start=True)
        self.time_d = {}
        fair_loss_list = list()
        logging.info('dyn_method: {}'.format(self.dyn_method))
        if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method:
            num_epoch = self.tepoch
            shuffle = False
            if 'nonseq' in self.dyn_method:
                shuffle = True
            if snap_idx == 0:
                num_epoch = self.epoch
                shuffle = True
        elif 'fulltrain' in self.dyn_method or 'pretrain' in self.dyn_method:
            num_epoch = self.epoch
            shuffle = True
            if 'seq' in self.dyn_method:
                shuffle = False

        recall_list = []
        cnt = 0
        best_recall = 0
        best_epoch = 0


        titer = tqdm(range(num_epoch), ncols=300)
        for epoch in titer:
            self._check_time()

            # if there is a pre-trained model, load it
            # if 'finetune' in self.dyn_method and os.path.exists(model.model_path+'_snap{}'.format(0)):
            #     print('Already trained: {}'.format(model.model_path+'_snap{}'.format(0)))
            #     model.load_model(add_path='_snap{}'.format(0))
            #     break

            loss, ori_loss, fair_loss, pd, flag = self.fit(model, data_dict, shuffle)
            training_time = self._check_time()

            titer.set_description('Epoch {:<3} loss={:<.4f} ori_loss={:<.4f} fair_loss={:<.4f} [{:<.1f} s] test_file: {} '.format(
                            epoch + 1, loss, ori_loss, fair_loss, training_time, args.test_result_file), refresh=True)

            # Print first and last loss/test
            # logging.info("Epoch {:<3} loss={:<.4f} ori_loss={:<.4f} fair_loss={:<.4f} [{:<.1f} s] ".format(
            #                 epoch + 1, loss, ori_loss, fair_loss, training_time))
            if flag:
                logging.info('NaN loss, stop training')
                break
            fair_loss_list.append(fair_loss)


            if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method:
                early_stop = 5
                patience = 10
                minimum = 5
                if snap_idx == 0:
                    early_stop = 100
                    patience = 10
                    minimum = 0
            elif 'fulltrain' in self.dyn_method or 'pretrain' in self.dyn_method:
                early_stop = 100
                patience = 10
                minimum = 20

            a = 0
            b = 0

            #if epoch >= 20 and (epoch+1) % 5 == 0:
            if epoch+1 >= minimum and (epoch+1) % 5 == 0:
                v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
                Inference.print_results(None, v_results, None)
                t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
                Inference.print_results(None, None, t_results)

                if v_results[a][b] > best_recall:
                    best_epoch = epoch+1
                    best_recall = v_results[a][b] # top-10 or top-20
                    best_v, best_t = v_results, t_results
                    model.save_model(add_path='_snap{}'.format(snap_idx))
                    #torch.save(Recmodel.state_dict(), weight_file)

                # if epoch == early_stop:
                #     recall_list.append((epoch, v_results[1][0]))
                # early stopping
                if epoch+1 > early_stop:
                    #recall_list.append((epoch, v_results[1][0]))
                    if v_results[a][b] < best_recall:
                        cnt += 1
                    else:
                        cnt = 1
                    if cnt >= patience:
                        break
        
        logging.info("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
        logging.info("Validation:")
        val_str = Inference.print_results(None, best_v, None)
        logging.info(val_str)
        val_result_filename_ = os.path.join(self.test_result_file, 'val_snap{}.txt'.format(snap_idx))
        open(val_result_filename_, 'w+').write(val_str)

        logging.info("Test:")
        test_str = Inference.print_results(None, None, best_t)
        logging.info(test_str)
        result_filename_ = os.path.join(self.test_result_file, 'test_snap{}.txt'.format(snap_idx))
        open(result_filename_, 'w+').write(test_str)


        return self.time[1] - self.time[0], best_epoch

    def fit(self, model, data, shuffle):

        gc.collect()
        torch.cuda.empty_cache()

        loss_lst, ori_loss_lst, fair_loss_lst = list(), list(), list()
        pd_list = list()
        # if 'finetune' in self.dyn_method:
        #     shuffle = False
        # else:
        #     shuffle = True
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, pin_memory=self.pin_memory)
        
        #for current in tqdm(dl, leave=True, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
        flag = 0
        for current in dl:
            #print('current: {}'.format(current['index']))
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            loss, prediction, ori_loss, fair_loss, pd = self.train_recommender_vanilla(model, current, data)

            loss_lst.append(loss)
            ori_loss_lst.append(ori_loss)
            if fair_loss is not None:
                fair_loss_lst.append(fair_loss)
            if pd is not None:
                pd_list.append(pd)

            flag = np.isnan(prediction).any()
            if flag: 
                break

        return np.mean(loss_lst).item(), np.mean(ori_loss_lst).item(), np.mean(fair_loss_lst).item(), np.mean(pd_list).item(), flag

    def train_recommender_vanilla(self, model, current, data):
        # Train recommender
        model.train()
        # Get recommender's prediction and loss from the ``current'' data at t
        prediction = model(current['user_id'], current['item_id'], self.DRM)
        loss, ori_loss, fair_loss, pd = model.loss(prediction, current, data, reduction='mean')

        # Update the recommender
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        # print('loss: {}'.format(loss))
        # print('model.optimizer: {}'.format(model.optimizer))

        if fair_loss is not None:
            fair_loss = fair_loss.cpu().data.numpy()
        if pd is not None:
            pd = pd.cpu().data.numpy()

        return loss.cpu().data.numpy(), prediction.cpu().data.numpy(), ori_loss.cpu().data.numpy(), fair_loss, pd

