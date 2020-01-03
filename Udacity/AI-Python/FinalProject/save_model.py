import torch
from torchvision import datasets

def save_checkpoint(model, arch, hidden_units, class_to_idx, save_dir):
    """
        Saves a pth file to the path designated on program run.
    
        Inputs: 
            model: trained model for which state should be saved
            arch: (str) chosen architecture
            hidden_units: (int) number of hidden units in hidden layer
            class_to_idx: (dictionary) relates flower class to an index
            save_dir: (str) path for pth save file
            
        Returns:
            Nothing. Saves the checkpoint to the designated directory.
            
    """
    
    ## define classifier state dict 
    if arch in ['vgg', 'densenet']:
        classifier_state_dict = model.classifier.state_dict()
    
    elif arch == 'resnet':
        classifier_state_dict = model.fc.state_dict()
    
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'classifier_state': classifier_state_dict,
                  'idx_to_class': {value: key for key, value in class_to_idx.items()}}
    
    
    torch.save(checkpoint, save_dir)
    
def save_test(save_dir):
    """ test the save functionality """
    checkpoint = {'arch': 'arch',
                  'hidden_units': 'hidden units',
                  'classifier_state': 'classifier_state_dict',
                  'idx_to_class': 'class stuff'}
    
    torch.save(checkpoint, save_dir)