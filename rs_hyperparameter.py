import numpy as np
from config import parser

np.random.seed(parser.seed)

search_size = 10
rs_tunes = 'learning_rate,rand_node_rate,beta,alpha,gamma'

# hps_dropout = [0] * 14
# hps_lr = [None] * search_size    # [0.00001, 0.00003, 0.00005]
# hps_lr = np.random.rand(search_size) * 0.004 + 0.001    # [0.001, 0.005]
hps_lr = np.random.rand(search_size)*2-6
hps_lr = np.power(10, hps_lr)   
hps_rand_node_rate = np.random.rand(search_size)*0.5
# hps_encoder = ['gae', 'gvae'] * 10
# hps_beta = [0.0001]
# hps_alpha = [0.0001] #[0.001, 0.0001, 0.00001, 0.000001]
hps_gamma = [None] * search_size #[0.0001, 0.00001, 0.000001]
hps_beta = [None] * search_size
hps_alpha = [None] * search_size
# hps_beta = np.random.rand(search_size)*4-4
# hps_beta = np.power(10, hps_beta)
# hps_alpha = np.random.rand(search_size)*4-4
# hps_alpha = np.power(10, hps_alpha)
# hps_gamma = np.random.rand(search_size)*3-1
# hps_gamma = np.power(10, hps_gamma)


rs_hp_range = {
    # "dropout": hps_dropout,
    "learning_rate": hps_lr,
    "rand_node_rate": hps_rand_node_rate,
    "beta": hps_beta,
    "alpha": hps_alpha,
    "gamma": hps_gamma,
    # "encoder": hps_encoder
}


def rs_set_hp_func(args, hp_values):
    hyperparams = rs_tunes.split(',')
    for hp in hyperparams:
        if hp_values[hp] is None: continue
        if hp == 'dropout':
            args.dropout = hp_values[hp]
        if hp == 'learning_rate':
            args.learning_rate = hp_values[hp]
        if hp == 'rand_node_rate':
            args.rand_node_rate = hp_values[hp]
        if hp == 'encoder':
            args.encoder = hp_values[hp]
        if hp == 'beta':
            args.beta = hp_values[hp]
        if hp == 'alpha':
            args.alpha = hp_values[hp]
        if hp == 'gamma':
            args.gamma = hp_values[hp]
