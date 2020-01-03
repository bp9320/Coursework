import torch
from torch import nn

from workspace_utils import keep_awake

def test_model(model, testloader, gpu):
    """
        test a model.
        Inputs:
            model: the trained model you want to test
            testloader: test data set
            gpu: (bool) does user want to use gpu if available
        
        Returns:
            Returns nothing. Prints test results on screen.
    """
    
    def get_device(gpu):
        """
            determines which device to use for training based on user input
            and availability of gpu
            
            Inputs:
                gpu: (bool) if user designated the gpu variable or not
            
            Returns:
                the device that should be used for training
        """
        
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        
        return device
    
    ## create criterion and optimizer
    criterion = nn.NLLLoss()
    
    ## define device and send model to it
    device = get_device(gpu)
    model.to(device)
    
    ## testing loop
    print(f'Starting testing on {"gpu" if device == "cuda" else device}.\n')
    
    ## set model to eval and initiate test variables
    model.eval()
    test_loss = 0
    accuracy = 0
    
    ## disable gradients
    with torch.no_grad():
        
        ## loop through testloader
        for images, labels in keep_awake(testloader):
                       
            ## Send test data to device
            images, labels = images.to(device), labels.to(device)
            
            ## run image thorugh model
            test_logps = model(images)
            test_batch_loss = criterion(test_logps, labels)
            
            test_loss += test_batch_loss.item()
            
            test_output = torch.exp(test_logps)

            ## compare prediction to label and calculate accuracy            
            top_p, top_class = test_output.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
    ## print testing loss and accuracy to screen        
    print(f'Test Loss: {test_loss/len(testloader):.3f}\n'
          f'Test Accuracy: {accuracy/len(testloader):.3f}\n'
         )