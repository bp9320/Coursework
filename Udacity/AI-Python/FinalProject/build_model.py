import torch
from torch import nn
from torchvision import models


## grad off function
def grad_off (model):
    """
    Turns off gradient for features so pretrained model does not get trained.
    input: model
    output: model with gradient turned off
    """
      
    for param in model.parameters():
        param.requires_grad = False
      
    return model

def build_training_model(arch, hidden_units):
    """
    Build the selected model for use in training or predictions
    inputs:
        arch: (str) the selected architecture (resnet, vgg, or densenet)
        hidden_layers: (int) number of elements in the hidden layer
    returns: pretrained model with designated architecture and hidden layers
    """
    
    print('*** Begin Building Training Model ***')
        
    if arch == 'resnet':
        ## retrieve pretrained model
        model = models.resnet50(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)

        ## define classifier for model
        classifier = nn.Sequential(nn.Linear(2048, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
        
        ## replace classifier
        model.fc = classifier
        
    elif arch == 'vgg':
        ## retrieve pretrained model
        model = models.vgg13(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)
        
        ## define classifier for model
        classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))        
        ## replace classifier
        model.classifier = classifier
    
    elif arch == 'densenet':
        ## retrieve pretrained model
        model = models.densenet161(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)
        
        ## define classifier for models
        classifier = nn.Sequential(nn.Linear(2208, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))        
        ## replace classifier
        model.classifier = classifier
        
    print('*** Building Training Model Completed ***')
        
    return model


## build model for prediction program from checkpoint

def build_predict_model(checkpoint_path):
    """
    Build the saved model for use in predictions
    inputs:
        checkpoint_path: (str) the path to the saved model (.pth)
    returns: tuple (previously trained model defind by .pth file, idx_to_class dict)
    """
    
    print('*** Begin Building Prediction Model ***')
    
    ## load in checkpoint
    
    ### Code courtesy of Tom Hale on Stack Overflow
    ### https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location = map_location)       
    
    ## build model from checkpoint
    if checkpoint['arch'] == 'resnet':
        ## retrieve pretrained model
        model = models.resnet50(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)

        ## define classifier for model
        classifier = nn.Sequential(nn.Linear(2048, checkpoint['hidden_units']),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(checkpoint['hidden_units'], 102),
                                   nn.LogSoftmax(dim=1))
        
        ## replace classifier
        model.fc = classifier
        
        ## load classifier state
        model.fc.load_state_dict(checkpoint['classifier_state'])
        
    elif checkpoint['arch'] == 'vgg':
        ## retrieve pretrained model
        model = models.vgg13(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)
        
        ## define classifier for model
        classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(checkpoint['hidden_units'], 102),
                                   nn.LogSoftmax(dim=1))        
        ## replace classifier
        model.classifier = classifier
        
        ## load classifier state
        model.classifier.load_state_dict(checkpoint['classifier_state'])
    
    elif checkpoint['arch'] == 'densenet':
        ## retrieve pretrained model
        model = models.densenet161(pretrained = True)
        
        ## turn off gradients
        model = grad_off(model)
        
        ## define classifier for models
        classifier = nn.Sequential(nn.Linear(2208, checkpoint['hidden_units']),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),
                                   nn.Linear(checkpoint['hidden_units'], 102),
                                   nn.LogSoftmax(dim=1))        
        ## replace classifier
        model.classifier = classifier
        
        ## load classifier state
        model.classifier.load_state_dict(checkpoint['classifier_state'])
        
        
    print('*** Building Prediction Model Completed ***')
        
    return (model, checkpoint['idx_to_class'])

# checkpoint = {'arch': arch,
#                   'hidden_units': hidden_units,
#                   'classifier_state': classifier_state_dict,
#                   'idx_to_class': {value: key for key, value in class_to_idx.items()}}