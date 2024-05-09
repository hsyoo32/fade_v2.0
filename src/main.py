# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import torch

from helpers import Reader, Runner
from models import  Model
from models.general import BPR, NCF, LightGCN
from models import Dataloader
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default='',
                        help='Result file path')
    parser.add_argument('--result_folder', type=str, default='',
                        help='Result folder path')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    parser.add_argument('--dyn_update', type=int, default=0,
                        help='dynamic update strategy.')
    return parser

def test_file(args, corpus, test_type):
    d = {}
    for idx in range(corpus.n_snapshots):
        val_result_filename_ = os.path.join(args.test_result_file, '{}_snap{}.txt'.format(test_type, idx))
        with open(val_result_filename_, 'r') as f:
            lines = f.readlines()
            data = [line.replace('\n', '').split() for line in lines]
            for value in data:
                if d.get(value[0]) is None:
                    d[value[0]] = []
                d[value[0]].append(float(value[1]))
    # print(d)

    with open(os.path.join(args.test_result_file, '_{}_mean'.format(test_type)), 'w+') as f:
        # for k, v in d.items():
        #     f.writelines('{}\t'.format(k))
        # f.writelines('\n')
        # for k, v in d.items():
        #     f.writelines('{:.4f}\t'.format(sum(v)/len(v)))
        
        for k, v in d.items():
            f.writelines('{}\t{:.4f}\n'.format(k, sum(v)/len(v)))
    
    with open(os.path.join(args.test_result_file, '_{}_trend'.format(test_type)), 'w+') as f:
        for k, v in d.items():
            f.writelines('{}'.format(k))
            for v_ in v:
                f.writelines('\t{:.4f}'.format(v_))
            f.writelines('\n')

def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)

    # Random seed
    utils.fix_seed(args.random_seed)

    # GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    logging.info('cuda device: {}'.format(args.gpu))

    # Read data
    # corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    corpus_path = os.path.join(args.path, args.dataset, args.suffix, args.s_fname, model_name.reader + '.pkl')
    
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
        #logging.info('Corpus loaded')
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    args.keys = ['train', 'test']
    logging.info('Total instances: {}'.format(corpus.dataset_size))
    logging.info('Train instances: {}'.format(corpus.n_train_batches))
    logging.info('Test instances: {}'.format(corpus.n_test_batches))
    logging.info('Snap boundaries: {}'.format(corpus.snap_boundaries + [corpus.dataset_size]))

    # Run model 
    runner = runner_name(args, corpus)
    data_dict = dict()
    force_train = True
    # pretrain = False

    # full-retraining   
    #utils.fix_seed(args.random_seed)
    if 'fulltrain' in args.dyn_method or 'pretrain' in args.dyn_method:
        data_type = 'hist'
    elif 'finetune' in args.dyn_method or 'newtrain' in args.dyn_method:
        data_type = 'incre'

    # if 'fulltrain' in args.dyn_method or 'finetune' in args.dyn_method:
        
    #phase = 'fulltrain'
    time_d = {}
    best_epoch_list = []
    #for idx, n_idx in enumerate(corpus.snap_boundaries): 
    for idx in range(corpus.n_snapshots):
        data_dict = Dataloader.Dataset(args, corpus, data_type, idx)

        utils.fix_seed(args.random_seed)
        model = model_name(args, corpus, data_dict)
        model.apply(model.init_weights)
        model.to(model._device)
        
        # pretrain model with base block
        if idx == 0 and force_train == False and os.path.exists(model.model_path+'_snap0'):
            continue

        # if model is already trained
        if os.path.exists(model.model_path+'_snap{}'.format(corpus.n_snapshots-1)):
            args.train = 0

        if args.train > 0 or force_train:
            # best_epoch
            t, best_epoch = runner.train(model, data_dict, args, corpus, snap_idx=idx)
            time_d['period_{}'.format(idx)] = t
            best_epoch_list.append(best_epoch)

        #print('snap_idx: {}'.format(idx))
            
    if args.train > 0 or force_train:
        with open(args.test_result_file+'_time_test', 'w+') as f:
            for k, v in time_d.items():
                f.writelines('{}\t'.format(k))
            f.writelines('\n')
            for k, v in time_d.items():
                f.writelines('{:.4f}\t'.format(v))
            f.writelines('\n')
            for k, v in time_d.items():
                f.writelines('{:.4f}\t'.format(v/60))
        
        with open(args.test_result_file+'_best_epoch', 'w+') as f:
            for k, v in enumerate(best_epoch_list):
                f.writelines('{}\t'.format(v))

    test_file(args, corpus, 'test')
    test_file(args, corpus, 'val')
        # d = {}
        # for idx in range(corpus.n_snapshots):
        #     val_result_filename_ = os.path.join(args.test_result_file, 'val_snap{}.txt'.format(idx))
        #     with open(val_result_filename_, 'w+') as f:
        #         lines = f.readlines()
        #         data = [line.replace('\n', '') for line in lines]
        #         for value in data:
        #             if d.get(value[0]) is None:
        #                 d[value[0]] = []
        #             d[value[0]].append(float(value[1]))
            
        # with open(os.path.join(args.test_result_file, '_{}_mean'.format(test_type)), 'w+') as f:
        #     for k, v in d.items():
        #         f.writelines('{}\t'.format(k))
        #     f.writelines('\n')
        #     for k, v in d.items():
        #         f.writelines('{:.4f}\t'.format(sum(v)/len(v)))
        
        # with open(os.path.join(args.test_result_file, '_{}_trend'.format(test_type)), 'w+') as f:
        #     for k, v in d.items():
        #         f.writelines('{}\t'.format(k))
        #         for v_ in v:
        #             f.writelines('{:.4f}\t'.format(v_))


    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)

def post():
    return args.test_result_file

if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_parser.add_argument('--dyn_method', type=str, default='default', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    #print(init_args.model_name)
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))
    #tester_name = eval('{}.{}'.format('Runner','Tester'))
    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    #parser = tester_name.parse_tester_args(parser)
    args, extras = parser.parse_known_args()

    if init_args.dyn_method == 'finetune':
        pass
    elif 'fulltrain' in init_args.dyn_method:
        args.tepoch = -1
    elif 'pretrain' in init_args.dyn_method:
        args.tepoch = -1

    if args.DRM == 'none':
        args.tau = -1
        args.num_neg_fair = -1
        args.DRM_weight = 1.0
        print('when DRM is none: no fairness reg')


    args.s_fname = '{}_{}_{}_s{}'.format(args.split_type, args.train_ratio, args.batch_size, args.n_snapshots)
    #log_args = [init_args.model_name, args.dataset, args.suffix, args.s_fname] # + str(args.test_length)]
    log_args1 = [args.dataset, args.s_fname, init_args.dyn_method] # + str(args.test_length)]
    log_args2 = []

    log_args = [args.dataset, args.s_fname, init_args.dyn_method]

    if args.DRM =='none':
        params = ['lr','l2','epoch', 'tepoch', 'num_neg', 'random_seed']
    else:
        params = ['lr','l2','epoch', 'tepoch', 'num_neg','num_neg_fair', 'DRM', 'DRM_weight', 'tau']
 
    for arg in params:
        log_args2.append(arg + '=' + str(eval('args.' + arg)))

    log_file_name1 = '__'.join(log_args1).replace(' ', '__')
    log_file_name2 = '__'.join(log_args2).replace(' ', '__')
    ### for test
    folder_name = init_args.model_name 

    if args.model_path == '':
        args.model_path = '../model/{}/{}/{}/{}'.format(folder_name, log_file_name1, log_file_name2, init_args.dyn_method)
    utils.check_dir(args.model_path)

    if args.log_file == '':
        args.log_file = '../log/{}/{}/{}.txt'.format(folder_name, log_file_name1, log_file_name2)
    utils.check_dir(args.log_file)

    if args.test_result_file == '':
        args.test_result_file = '../test_result/{}/{}/{}/'.format(folder_name, log_file_name1, log_file_name2)
    utils.check_dir(args.test_result_file)
    
    args.dyn_method = init_args.dyn_method
    args.model_name = init_args.model_name
    
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(log_file_name1+'__'+log_file_name2)
    main()