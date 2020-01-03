from get_inputs import get_training_inputs, check_pth_file
from build_model import build_training_model
from data_prep import get_training_loaders
from train_new_model import train_model
from save_model import save_checkpoint, save_test
from test_model import test_model

def main():
    """
    Main function
    """
    
    ## get input arguments
    input_args = get_training_inputs()
#     print(input_args)

    ## validate save dir
    input_args.save_dir = check_pth_file(input_args.save_dir)
#     print(input_args)
    
    ## build model
    model = build_training_model(input_args.arch, input_args.hidden_units)
#     print(model)

    ## load data
    trainloader, validloader, testloader, class_to_idx = get_training_loaders(input_args.data_dir)
#     print(trainloader, validloader, testloader)
    
    ## train the model
    train_model(model, input_args.arch, trainloader, validloader, input_args.gpu, input_args.epochs, input_args.learn_rate)

    ## save model
    save_checkpoint(model, input_args.arch, input_args.hidden_units, class_to_idx, input_args.save_dir)
#     save_test(input_args.save_dir)
    
    ## test the model
    test_model(model, testloader, input_args.gpu)
    
    

if __name__ == '__main__':
    main()