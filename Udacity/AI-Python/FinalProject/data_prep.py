import torch
from torchvision import datasets, transforms
from PIL import Image

## define normalization means and stdevs
means = [0.485, 0.456, 0.406]
stdevs = [0.229, 0.224, 0.225]

def get_training_loaders (data_dir):
    """
        Builds the training and validation dataloaders along with class to index list
        inputs:
            data_dir: (str) path to directory with training/validation/test data
        returns:
            tuple (trainloader, validloader, testloader, class_to_idx)
    """
    
    print('*** Start Generating Dataloaders ***')
    
    ## define directory paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    ## define transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(45),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stdevs)])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stdevs)])
    
    ## create datasets
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    ## create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print('*** Dataloader Generation Complete ***')
    
    return (trainloader, validloader, testloader, train_data.class_to_idx)

## Prepare images for predction
def process_image(image):
    """
        transform input image to appropriate format
        inputs:
            image: (str) path to image
        returns:
            img_tensor
    """
    print('*** Start Processing Image ***')
    
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(means, stdevs)])
    
    img_pil = Image.open(image)
    img_tensor = process(img_pil)
    
    print('*** Processing Image Complete ***')
    
    return img_tensor
                                 
                                