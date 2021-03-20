import argparse

basedir = './dataset/faces-spring-2020/faces-spring-2020/'

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()
    # Base Directory
    parser.add_argument('-d', '--dataset', dest='dataset', default=basedir, 
                        help="Path to the dataset directory.")
    parser.add_argument('-m', '--model_save_path', type=str, default='./models/',
                        help="Path to Saved model directory.")
    # Selcet subject
    parser.add_argument('-s', '--subject', dest='subject', type=int, default=0, 
                        help="If colorize sketch select 0 otherwise 1")
    # Image size
    parser.add_argument('--height', dest='height', default=256, type=int,
                        help="Height of the image")
    parser.add_argument('--width', dest='width', default=256, type=int,
                        help="Width of the image")

    # Misc
    parser.add_argument('-t','--train', type=str2bool, default=True,help="True when train the model, else used for validation.")
    parser.add_argument('-v','--validate', default=None, type=str, help="Path to the image for validation through model prediction.")

    # Model hyperparameter
    parser.add_argument('-b', '--batch', dest='batch',type=int, default=1)
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-b1', '--beta1', type=float, default=0.0)
    parser.add_argument('-b2', '--beta2', type=float, default=0.9)
    parser.add_argument('-g_lr', '--gen_lr', type=float, default=0.0001)
    parser.add_argument('-d_lr', '--dis_lr', type=float, default=0.0004)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-9)
    parser.add_argument('-e', '--epoch', type=int, default=800)

    return parser.parse_args()