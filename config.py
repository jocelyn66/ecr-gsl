import argparse


# data configuration#####################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "--d", type=str, default="ecb+",
                    help="Knowledge Graph dataset")
parser.add_argument("--dataset_root_path", type=str, default="./data/ECB+")
parser.add_argument('--train_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Train_processed_data.json')
parser.add_argument('--dev_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Dev_processed_data.json')
parser.add_argument('--test_file', type=str, default='./data/ECB+/processed_ECB+/ECB_Test_processed_data.json')
parser.add_argument('--schema_path', type=str, default="./data/ECB+/processed_ECB+/ECB_schema.json")
parser.add_argument('--ckp_dir', type=str, default=None)
parser.add_argument('--train_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_train.arrow')
parser.add_argument('--dev_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_dev.arrow')
parser.add_argument('--test_cache_file', type=str, default='./data/ECB+/dataset_cache/ECB_test.arrow')
parser.add_argument('--log_dir', type=str, default=None)              

parser.add_argument("--gpu", type=int, default=0,
                    help="gpu")
parser.add_argument("--debug", action="store_true",
                    help="Only use 1000 examples for debugging")
parser.add_argument("--double-precision", action="store_true",
                    help="Machine precision")
parser.add_argument("--save-freq", type=int, default=1000)
parser.add_argument("--seed", type=int, default=22)

# training configuration###############

parser.add_argument("--optimizer", choices=["Adagrad", "Adam"], default="Adam",
                    help="Optimizer")
parser.add_argument("--regularizer", choices=[None, "N3", "F2"], default=None,
                    help="Regularizer")
parser.add_argument("--reg", default=0, type=float,
                    help="Regularization weight")   # 0: no reg

parser.add_argument("--max-epochs", default=100, type=int,
                    help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=500, type=int,
                    help="Number of epochs before early stopping")
parser.add_argument("--valid-freq", default=1, type=int,
                    help="Number of epochs before validation")
parser.add_argument("--batch-size", default=512, type=int,
                    help="Batch size")

parser.add_argument("--init-size", default=1e-3, type=float,
                    help="Initial embeddings' scale")
parser.add_argument("--learning-rate", default=1e-5, type=float,
                    help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-2)

# regularization
parser.add_argument("--beta", type=float, default=0., help="weight of nuclear norm")
parser.add_argument("--alpha", type=float, default=0., help="weight of L1 norm")
parser.add_argument("--gamma", type=float, default=0., help="weight of F norm")
parser.add_argument("--loss-type", "--loss", type=int, default=0, choices=[0, 1, 2 , 3, 4], 
                    help="type of objective function, 0: no reg, 1: L = CE + nuclear, 2: L = CE + nuclear + L1 + F, 3: L = CE(A,A') + nuclear + L1, 4: L = CE(A,A+H) + nuclear + L1")

# configuration for optimal parameters
parser.add_argument("--rand-search", "--rs", action='store_true',
                    help="perform random search for best configuration")
parser.add_argument("--grid-search", "--gs", action='store_true',
                    help="perform grid search for best configuration")
# mdoel config
parser.add_argument("--model", default="ecr-gsl",
                    help="ECR models")
parser.add_argument('--tokenizer-name', type=str, default='bert-base-uncased')
parser.add_argument('--plm-name', type=str, default='bert-base-uncased')
parser.add_argument('--feat-dim', type=int, default=768, 
                    help='size of features, i.e. bert embedding dim')
parser.add_argument("--fine-tune", action='store_true')
parser.add_argument("--cls", action='store_true', help='concat [CLS] embedding')
parser.add_argument("--win-w", type=float, default=0., help='weight applyed to token out of span, w=1 for token in span')
parser.add_argument("--win-len", type=int, default=0, help='')
parser.add_argument("--win-max-pool", action='store_true', help='max pooling or mean pooling')

# encoder config#####################
parser.add_argument('--rand-node-rate', type=float, default=0.2)
parser.add_argument("--encoder", type=str, choices = ['gae', 'gvae'], default='gae',
                    help="gsl model")
parser.add_argument("--dropout", "--en-dropout", type=float, default=0.,
                    help="dropout probability")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')

# decoder
parser.add_argument("--decoder", type=str, default='')  #无用
# parser.add_argument('--threshold', type=float, default=0.5)

parser = parser.parse_args()
