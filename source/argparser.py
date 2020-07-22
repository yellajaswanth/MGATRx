import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=2000,
                        help='Number of epochs to train. Default is 100.')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Default learning rate is 0.05.')

    parser.add_argument('--embed-size',
                        type=int,
                        default=512,
                        help='The embedding size of each view. Default is 128.')

    parser.add_argument('--weight-decay',
                        type=float,
                        default=0,
                        help='Weight decay (L2 loss on parameters). Default is 5e-4.')


    parser.add_argument('--encoder-activation',
                        type=str,
                        default='selu',
                        choices=['leaky','selu', 'relu','prelu','tanh','sigmoid','elu','none'],
                        help='Choose activation function for encoder. Default is `selu`')


    parser.add_argument('--save-model',
                        type=bool,
                        default=False,
                        choices=[True, False],
                        help='Do you want to save model for each fold? Model will be saved in `tmp/experiments/` folder. Default is `False`')

    parser.add_argument('--fold-test',
                        type=bool,
                        default=False,
                        choices=[True, False],
                        help='Are you testing the fold. Only first fold will be running. Default is `False`')


    parser.add_argument('--kfolds',
                        type=int,
                        default=10,
                        help='Number of folds for validation. Default is 10')


    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout Rate. Default is 0.5')

    parser.add_argument('--encoder',
                        type=str,
                        default='GAT',
                        choices=['GCN', 'GAT'],
                        help='Choose decoder type type. Default is `GCN`')

    parser.add_argument('--decoder',
                       type=str,
                       default='linear',
                       choices=['linear', 'bilinear', 'distmult', 'dedicom','innerproduct','mlp' ],
                       help='Choose decoder type type. Default is `linear`')

    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed for consistency. Default is 123')

    parser.add_argument('--num-layers',
                        type=int,
                        default=1,
                        help='Number of layers in encoder. Default is 1')

    parser.add_argument('--valid-size',
                        type=float,
                        default=0.15,
                        help='Validation set size. Default is 0.15')



    # parser.add_argument('--mlflow-address',
    #                     type=str,
    #                     default='10.200.42.157',
    #                     choices=['localhost','10.200.42.157'],
    #                     help='Choose dataset. Default is `drugbank`')
    #
    # parser.add_argument('--mlflow-enable',
    #                     type=bool,
    #                     default=True,
    #                     choices=[True, False],
    #                     help='Enable mlflow tracking. Default is `True`')

    return parser.parse_args()

