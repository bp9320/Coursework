import argparse

## get inputs for train program
def get_training_inputs():
    """
    Defines the inputs for the training program.
    Arguments:
        data_dir: (string) required, defines the directory for the training data
        save_dir: (string) optional, defines save file name for the trained model's state dict. default is "temp.pth" and will be                   overwritten by subsequent trainings with the default setting.
        arch: (string) architechture choice. valid options are resnet, vgg, and densenet
    """
    
    ## Create parser and define arguments
    parser = argparse.ArgumentParser()
    
    ## training data directory
    parser.add_argument('--data_dir', type = str, default = '/home/workspace/ImageClassifier/flowers', 
                        help = 'path to parent directory for training data')
    
    ## save directory
    parser.add_argument('--save_dir', type = str, default = '/home/workspace/ImageClassifier/TrainedModels/temp.pth', 
                        help = 'path to file you want to save the trained model to')
    
    ## architecture choice
    parser.add_argument('--arch', type = str, default = 'resnet', choices=['resnet', 'vgg', 'densenet'],
                       help = 'choice of architecture for the model')
    
    ## learning rate
    parser.add_argument('--learn_rate', type = float, default = 0.001,
                        help = 'learning rate (float)')
    
    ## hidden units
    parser.add_argument('--hidden_units', type = int, default = 400,
                        help = 'number of units in hidden layer (int)')
    
    ## epochs
    parser.add_argument('--epochs', type = int, default = 10,
                        help = 'number of ephochs (int)')
    
    ## train on GPU
    parser.add_argument('--gpu', action = 'store_true', default = False, #argparse.SUPPRESS
                        help = 'train on GPU')
    
    return parser.parse_args()

## get inputs for predict program
def get_predict_inputs():
    """
    Defines the inputs for the prediction program.
    Arguments:
        img_path: (string) required, defines the file path for the image to predict
        checkpoint: (string) required, defines the file path for the saved model
        category_names: (string) optional, path to json file with category names
        top_k: (int) optional, determines how many predictions are returned
        gpu: optional, determines if predictions are run on cpu or gpu
    """
    
    ## Create parser and define arguments
    parser = argparse.ArgumentParser()
    
    ## image path
    parser.add_argument('img_path', type = str, 
                        help = 'path to parent directory for training data')
    
    ## checkpoint directory
    parser.add_argument('checkpoint', type = str,
                        help = 'path to .pth file of the trained model')
    
    ## category names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                       help = 'path to category names json file')
    
    ## top k
    parser.add_argument('--top_k', type = int, default = 1,
                        help = 'number of results to display (int)')
    
    ## train on GPU
    parser.add_argument('--gpu', action = 'store_true', default = False,
                        help = 'predict on GPU')
    
    return parser.parse_args()

## check pth files
def check_pth_file(pth_dir):
    """
        Check that the pth file ends with .pth and ask for new path if it does not
        Inputs:
            pth_dir: (str) user provided pth file
        Returns:
            pth_dir: (str) verified pth file path
    """
    
    if pth_dir.endswith('.pth'):
        return pth_dir
    else:
        while not pth_dir.endswith('.pth'):
            pth_dir = input("\nEnter file path that ends with '.pth' for model save file:  ")
        return pth_dir

## check json files
def check_cat_file(cat_path):
    """
        Check that category path is a json file
        Inputs:
            cat_path: (str) path to category file
        Returns:
            cat_path: (str) verified category file path
    """
    
    if cat_path.endswith('.json'):
        return cat_path
    else:
        while not cat_path.endswith('.json'):
            cat_path = input("\nEnter file path that ends with '.json' for category names:  ")
        return cat_path