import json
import logging
import os
import sys
import time

import torch.optim
import math
import itertools

import models.ecr_model
import optimizers.regularizers as regularizers
from optimizers.ecr_optimizer import *
from config import parser
from utils.name2object import name2model
from rs_hyperparameter import rs_tunes, rs_hp_range, rs_set_hp_func
from gs_hyperparameter import gs_tunes, gs_hp_range, gs_set_hp_func
from utils.train import *
from utils.eval import *
from utils.visual import *
from dataset.dataset_process import preprocess_function
from models.clustring import *
from models.ecr_model import ECRModel, ECRModel_fine_tune
from models.feature import *

from dataset.graph_dataset import GDataset, get_examples_indices
from datasets import load_dataset
from utils.bcubed_scorer import bcubed

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertModel
)


def set_logger(args):
    save_dir = get_savedir(args.dataset, args.model, args.encoder, args.decoder, args.rand_search or args.grid_search)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.logs")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    print("Saving logs in: {}".format(save_dir))
    return save_dir


def train(args, hps=None, set_hp=None, save_dir=None, num=-1, threshold=0.99):

    # config
    start_model = datetime.datetime.now()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.rand_search or args.grid_search:
        set_hp(args, hps)

    if not (args.rand_search or args.grid_search):
        save_dir = set_logger(args)
        with open(os.path.join(save_dir, "config.json"), 'a') as fjson:
            json.dump(vars(args), fjson)

    model_name = "model{}_feat-d{}_h1-d{}_h2-d{}.pt".format(num, args.feat_dim, args.hidden1, args.hidden2)
    logging.info(args)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)
        print("double precision")
    else:
        torch.set_default_dtype(torch.float32)

    # dataset###############
    dataset = GDataset(args)
    # for split in ['Train', 'Dev', 'Test']:
    #     assert(dataset.adjacency[split].diagonal(offset=0, axis1=0, axis2=1).all()==0)

    args.n_nodes = dataset.n_nodes

    # for split in ['Train', 'Dev', 'Test']:
    #     print("###", dataset.event_chain_list[split])

    # Some preprocessing:
    # adj_norm = preprocess_adjacency(adj_train)
    pos_weight = {}
    norm = {}
    # adj_norm = {}

    for split in ['Train', 'Dev', 'Test']:
        
        if not args.double_precision:
            for split in ['Train']:
                for s in ['sent', 'doc', 'event_coref', 'entity_coref']:
                    dataset.adjacency[split][s] = dataset.adjacency[split][s].astype(np.float32)
            for split in ['Dev', 'Test']:
                for s in ['sent', 'doc']:
                    dataset.adjacency[split][s] = dataset.adjacency[split][s].astype(np.float32)
        # adj = dataset.adjacency[split]
        # adj_norm[split] = preprocess_adjacency(adj)

    # process train adj for loss cal
    n_edges_dict = {}
    #dtype?
    adj_label = np.zeros((args.n_nodes['Train'], args.n_nodes['Train']))
    for s in ['sent', 'doc', 'event_coref', 'entity_coref']:
        adj_label = np.where(adj_label>0, adj_label, dataset.adjacency['Train'][s].toarray())
        n_edges_dict[s] = dataset.adjacency['Train'][s].sum() - args.n_nodes['Train']
    norm = adj_label.shape[0] * adj_label.shape[0] / float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2)
    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    pos_weight = torch.tensor([pos_weight])

    event_true_sub_indices = {}
    event_false_sub_indices = {}
    entity_true_sub_indices = {}
    entity_false_sub_indices = {}
    recover_true_sub_indices = {}
    recover_false_sub_indices = {}

    #process for lp eval
    #重构label不取对角线
    for split in ['Train', 'Dev', 'Test']:
        if split in ['Train']:
            adj = adj_label
        else:
            adj = dataset.adjacency[split]['doc']  #dev, test:句子 文档关系, 满足句子一定满足文档
        event_true_sub_indices[split], event_false_sub_indices[split] = get_examples_indices(adj, dataset.event_idx[split])
        entity_true_sub_indices[split], entity_false_sub_indices[split] = get_examples_indices(adj, dataset.entity_idx[split])
        recover_true_sub_indices[split], recover_false_sub_indices[split] = get_examples_indices(adj, list(range(args.n_nodes[split])))

    adj_label += np.eye(adj_label.shape[0], dtype=adj_label.dtype)
    adj_label = torch.tensor(adj_label)

    #Load Schema
    with open(args.schema_path, 'r') as f:  #3个set的schema
        schema_list = json.load(f)
        doc_schema = schema_list[0]
        event_schema = schema_list[1]
        entity_schema = schema_list[2]
    # bert################################
     #Load Datasets
    data_files = {}
    data_files["train"] = args.train_file
    data_files["dev"] = args.dev_file
    data_files["test"] = args.test_file
    datasets = load_dataset("json", data_files=data_files)

    #introduce PLM
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    plm = BertModel.from_pretrained(args.plm_name)

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.train_cache_file
    )

    dev_dataset = datasets["dev"]
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.dev_cache_file
    )

    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.test_cache_file
    )

    dataset_list = {'Train':train_dataset, 'Dev':dev_dataset, 'Test':test_dataset}
    ######################

    # create 
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!")
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda" if use_cuda else "cpu")

    #get features
    features_list = get_bert_features(plm, dataset_list, args.device)
    for split in ['Train', 'Dev', 'Test']:
    #     torch.save(features_list[split], os.path.join(save_dir, 'bert_features_{}.pt'.format(split)))
        features_list[split] = features_list[split].to('cpu')
    torch.cuda.empty_cache()

    if use_cuda:
        for split in ['Train', 'Dev', 'Test']:
            features_list[split] = features_list[split].to(args.device)
        adj_label = adj_label.to(args.device)
        pos_weight = pos_weight.to(args.device)

    if args.fine_tune:
        model = ECRModel_fine_tune(args, tokenizer, plm, schema_list)
    else:
        model = ECRModel(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(args.device)    # GUP

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # regularizer = None
    # if args.regularizer:
    #     regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optimizer = GAEOptimizer(args, model, optim_method, use_cuda, dataset.adjacency['Train'], n_edges_dict, pos_weight)

    # start train######################################
    counter = 0
    best_f1 = None
    best_epoch = None
    best_model_path = ''
    hidden_emb = None
    losses = {'Train': [], 'Dev': [], 'Test': []}
    b3s = {'Train': [], 'Dev': [], 'Test': []}
    nmis = {'Train': [], 'Dev': [], 'Test': []}
    stats = {}

    logging.info("\t ---------------------------Start Optimization-------------------------------")
    for epoch in range(args.max_epochs):
        t = time.time()
        model.train()
        if use_cuda:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # loss, mu = optimizer.epoch(train_dataset, adj_norm['Train'], dataset.adjacency['Train'])
        loss, mu = optimizer.epoch(features_list['Train'], dataset.adjacency['Train'], adj_label, norm, pos_weight)
        losses['Train'].append(loss)
        logging.info("Epoch {} | ".format(epoch))
        logging.info("\tTrain")
        logging.info("\t\taverage train loss: {:.4f}".format(loss))
        if math.isnan(loss):
            break
            
        logging.info("\ttraining time={:.5f}".format(time.time() - t))
        t = time.time()

        # valid training set
        hidden_emb = mu.data.detach().cpu().numpy()
        
        model.eval()

        #AUC, AP###############
        # metrics1 = test_model(hidden_emb, dataset.event_idx['Train'], event_true_sub_indices['Train'], event_false_sub_indices['Train'])
        # logging.info("\t\tevent coref:" + format_metrics(metrics1, 'Train'))

        # entity_idx = list(set(range(args.n_nodes['Train'])) - set(dataset.event_idx['Train']))
        # metrics2 = test_model(hidden_emb, entity_idx, entity_true_sub_indices['Train'], entity_false_sub_indices['Train'])
        # logging.info("\t\tentity coref:" + format_metrics(metrics2, 'Train'))

        # metrics3 = test_model(hidden_emb, list(range(args.n_nodes['Train'])), recover_true_sub_indices['Train'], recover_false_sub_indices['Train'])
        # logging.info("\t\treconstruct adj:" + format_metrics(metrics3, 'Train'))

        # B3###################
        # val#####################################
        # 无监督
        if (epoch+1) % args.valid_freq == 0:
            
            for split in ['Train', 'Dev', 'Test']:

                if split in ['Dev', 'Test']:

                    loss, mu = optimizer.eval(features_list[split], dataset.adjacency[split])  # norm adj
                    losses[split].append(loss)
                    logging.info("\t{}".format(split))
                    logging.info("\t\taverage {} loss: {:.4f}".format(split, loss))

                hidden_emb = mu.data.detach().cpu().numpy()

                logging.info("\tEvaluate Link Prediction:")
                #auc, ap: test event mention pair###########
                test_metrics1 = test_model(hidden_emb, dataset.event_idx[split], event_true_sub_indices[split], event_false_sub_indices[split])
                logging.info("\t\tevent coref:" + format_metrics(test_metrics1, split))

                test_metrics2 = test_model(hidden_emb, dataset.entity_idx[split], entity_true_sub_indices[split], entity_false_sub_indices[split])
                logging.info("\t\tentity coref:" + format_metrics(test_metrics2, split))

                test_metrics3 = test_model(hidden_emb, list(range(args.n_nodes[split])), recover_true_sub_indices[split], recover_false_sub_indices[split])
                logging.info("\t\treconstruct adj:" + format_metrics(test_metrics3, split))

                #b3###########################
                logging.info("\tB3 Evaluation in {}:".format(split))
                for threshold in [0.95, 0.9, 0.85]:
                    # eval_model_leiden of eval_model_louvain
                    logging.info("\t\tevent coref:")
                    pred_list, n_comm, n_edges = eval_model_leiden(save_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
                    logging.info("\t\t{}, n_edges={}".format(threshold, n_edges))
                    logging.info("\t\tlouvain: n_community = {}".format(n_comm))
                    
                    eval_metrics = bcubed(dataset.event_chain_list[split], pred_list)
                    nmi_metric = cal_nmi(dataset.event_chain_list[split], pred_list)
                    logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                    logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))
                    add_new_item(stats, 'b3_r_'+str(threshold), eval_metrics[0], split)
                    add_new_item(stats, 'b3_p_'+str(threshold), eval_metrics[1], split)
                    add_new_item(stats, 'b3_f_'+str(threshold), eval_metrics[2], split)
                    add_new_item(stats,'nmi_'+str(threshold), nmi_metric, split)

                    # pred_list2, n_comm2, n_edges = eval_model_louvain(save_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
                    # logging.info("\t\tleiden: n_community = {}".format(n_comm2))
                    # eval_metrics2 = bcubed(dataset.event_chain_list[split], pred_list2)
                    # logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics2))

                    logging.info("\t\tentity coref:")
                    pred_list, n_comm, n_edges = eval_model_leiden(save_dir, split, hidden_emb, dataset.entity_idx[split], threshold, num)
                    logging.info("\t\tthreshold={}, n_edges={}".format(threshold, n_edges))
                    logging.info("\t\t\tlouvain: n_community = {}".format(n_comm))
                    
                    eval_metrics = bcubed(dataset.entity_chain_list[split], pred_list)
                    nmi_metric = cal_nmi(dataset.entity_chain_list[split], pred_list)
                    logging.info("\t\t\tb3 metrics:" + format_b3_metrics(eval_metrics))
                    logging.info("\t\t\tnmi={:.5f}".format(nmi_metric))
                    add_new_item(stats, 'ent_b3_r_'+str(threshold), eval_metrics[0], split)
                    add_new_item(stats, 'ent_b3_p_'+str(threshold), eval_metrics[1], split)
                    add_new_item(stats, 'ent_b3_f_'+str(threshold), eval_metrics[2], split)
                    add_new_item(stats,'ent_nmi_'+str(threshold), nmi_metric, split)

        logging.info("\t\tevaluation time={:.5f}".format(time.time() - t))
        # # 有监督
        # model.eval()
        # if (epoch + 1) % args.valid_freq == 0:
        #     # valid loss
        #     # valid metircs
        #     metrics = test_model()   # F1
        #     logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))
        #     logging.info(format_conll('Valid_F1'+valid_f1))  
        #     if not best_f1 or valid_f1 > best_f1:
        #         best_f1 = valid_f1
        #         counter = 0
        #         best_epoch = epoch
        #         logging.info("\t Saving model at epoch {} in {}".format(epoch, save_dir))
        #         best_model_path = os.path.join(save_dir, '{}_{}'.format(epoch, model_name))
        #         torch.save(model.cpu().state_dict(), best_model_path)
        #         if use_cuda:
        #             model.cuda()

        #     else:
        #         counter += 1
        #         if counter == args.patience:
        #             logging.info("\t Early stopping")
        #             break
        #         elif counter == args.patience // 2:
        #             pass
        # ###################

        #save_freq###########
        # if (epoch+1) % args.save_freq == 0 or (epoch + 1)==args.max_epochs:

        #     model_path = os.path.join(save_dir, str(epoch+1)+model_name)
            
        #     save_check_point(model, model_path)
        #     # torch.save(model.cpu().state_dict(), model_path)
        #     # model.to(args.device)

    logging.info("\t ---------------------------Optimization finished---------------------------")

    # # test#########################
    # if not best_f1:
    #     best_model_path = os.path.join(save_dir, model_name)
    #     torch.save(model.cpu().state_dict(), best_model_path)
    # else:
    #     logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
    #     model.load_state_dict(torch.load(best_model_path))  # load best model
    # if use_cuda:
    #     model.cuda()
    # model.eval()  # no BatchNormalization Dropout

    # # Test metrics

    # 测评
    # logging.info("Evaluation Test Set:")
    # test_f1 = None
    # conll_f1 = run_conll_scorer(args.output_dir)
    # logging.info(conll_f1)

    model_path = os.path.join(save_dir, model_name)  #最后一个epoch
    save_check_point(model, model_path)
    # torch.save(model.cpu().state_dict(), model_path)

    plot(save_dir, 'converg', num, losses['Train'], losses['Dev'], losses['Test'])

    str_metrics = ['b3_r','b3_p','b3_f','nmi']
    str_threshs = ['0.95','0.9','0.85']
    descrips = itertools.product(str_metrics, str_threshs)
    for des in descrips:
        s = des[0] + '_' + des[1]
        plot_splits(save_dir, s, num, stats[s])
        plot_splits(save_dir, 'ent_'+s, num, stats['ent_'+s])
    # plot1(save_dir, 'converg', num, losses['Train'])

    # # save statistics to file
    # res = []
    # for vdict in [losses, b3s, nmis]:
    #     if len(vdict['Test'])> 0 :
    #         res.append(vdict)
    #         plot(save_dir, 'converg', num, vdict['Train'], vdict['Dev'], vdict['Test'])

    # save statistics to file
    stat_dir = os.path.join(save_dir, str(num)+'statistics'+'.json')
    with open(stat_dir, 'w') as f:
        json.dump(stats, f)
        f.close()

    end_model = datetime.datetime.now()
    logging.info('this model runtime: %s' % str(end_model - start_model))
    logging.info("\t ---------------------------done---------------------------")
    return None


def rand_search(args):

    best_f1 = 0
    best_hps = []
    best_f1s = []

    save_dir = set_logger(args)
    logging.info("** Random Search **")

    args.tune = rs_tunes
    logging.info(rs_hp_range)
    hyperparams = args.tune.split(',')

    if args.tune == '' or len(hyperparams) < 1:
        logging.info("No hyperparameter specified.")
        sys.exit(0)
    grid = rs_hp_range[hyperparams[0]]
    for hp in hyperparams[1:]:
        grid = zip(grid, rs_hp_range[hp])

    grid = list(grid)
    logging.info('* {} hyperparameter combinations to try'.format(len(grid)))

    for i, grid_entry in enumerate(list(grid)):
        if not (type(grid_entry) is list):
            grid_entry = [grid_entry]
        grid_entry = flatten(grid_entry)    # list
        hp_values = dict(zip(hyperparams, grid_entry))
        logging.info('* Hyperparameter Set {}:'.format(i))
        logging.info(hp_values)

        test_metrics = train(args, hp_values, rs_set_hp_func, save_dir, i)
        logging.info('{} done'.format(grid_entry))
    #     if test_metrics['F'] > best_f1:
    #         best_f1 = test_metrics['F']
    #         best_f1s.append(best_f1)
    #         best_hps.append(grid_entry)
    # logging.info("best hyperparameters: {}".format(best_hps))


def grid_search(args):

    best_f1 = 0
    best_hps = []
    best_f1s = []

    save_dir = set_logger(args)
    logging.info("** Grid Search **")

    args.tune = gs_tunes
    logging.info(gs_hp_range)
    hyperparams = args.tune.split(',')

    if args.tune == '' or len(hyperparams) < 1:
        logging.info("No hyperparameter specified.")
        sys.exit(0)
    grid = gs_hp_range[hyperparams[0]]
    for hp in hyperparams[1:]:
        grid = itertools.product(grid, gs_hp_range[hp])

    grid = list(grid)
    logging.info('* {} hyperparameter combinations to try'.format(len(grid)))

    for i, grid_entry in enumerate(list(grid)):
        if not (type(grid_entry) is list):
            grid_entry = [grid_entry]
        grid_entry = flatten(grid_entry)    # list
        hp_values = dict(zip(hyperparams, grid_entry))
        logging.info('* Hyperparameter Set {}:'.format(i))
        logging.info(hp_values)

        test_metrics = train(args, hp_values, gs_set_hp_func, save_dir, i)
        logging.info('{} done'.format(grid_entry))
    #     if test_metrics['F'] > best_f1:
    #         best_f1 = test_metrics['F']
    #         best_f1s.append(best_f1)
    #         best_hps.append(grid_entry)
    # logging.info("best hyperparameters: {}".format(best_hps))


if __name__ == "__main__":
    start = datetime.datetime.now()
    if parser.rand_search:
        rand_search(parser)
    else:
        if parser.grid_search:
            grid_search(parser)
        else:
            train(parser)
    end = datetime.datetime.now()
    logging.info('total runtime: %s' % str(end - start))
    sys.exit()
