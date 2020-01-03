import torch
from torch import optim, nn

from workspace_utils import keep_awake

def train_model(model, arch, trainloader, validloader, gpu, epochs, learn_rate):
    """
        Train a model.
        Inputs:
            model: the model you want to train
            arch: (str) architecture of the model to create optimizer
            testloader: test data set
            validloader: validation data set
            gpu: (bool) does user want to use gpu if available
            epochs: (int) number of training epochs
            learn_rate: (float) learn rate for training
        
        Returns:
            trained model
    """
    
    def create_optimizer(model, arch, learn_rate):
        """
            function to create optimizer for model due to classifier key varying by model
            Inputs:
                model: model to train for access to parameters
                arch: (str) architecture of the model
                learn_rate: (float) learn rate for training
            Returns:
                Optimzer for the model
        """
        
        if arch in ['vgg', 'densenet']:
            optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
        
        elif arch == 'resnet':
            optimizer = optim.Adam(model.fc.parameters(), lr = learn_rate)
        
        return optimizer
    
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
    optimizer = create_optimizer(model, arch, learn_rate)
    
    ## define device and send model to it
    device = get_device(gpu)
    model.to(device)
    
    ## define string for training summary
    training_summary = '\n*** Training Summary ***\n'
    
    ## training loop
    print(f'Starting training on {"gpu" if device == "cuda" else device}.\n'
          f'You may want to grab a book. This may take a while...\n')
    for e in keep_awake(range(epochs)):
        
        print(f'Begin epoch {e+1}\n')
        
        step = 0
        
        ## ensure model is in training mode
        model.train()
        
        ## initiate running loss
        running_loss = 0
        
        print(f'\tBeginning training for epoch {e+1}\n')
        
        ## loop through trainloader
        for images, labels in trainloader:
            
            step += 1
            if step == 1 or step % 10 == 0:
                print(f'\t\tBeginning training step {step}')
            
            ## send images and labels to device
            images, labels = images.to(device), labels.to(device)
            
            ## reset optimizer
            optimizer.zero_grad()
            
            ## run training step
            train_logps = model(images)
            train_loss = criterion(train_logps, labels)
            train_loss.backward()
            optimizer.step()
            
            running_loss += train_loss.item()
        
        ## validate at end of each epoch
        
        ## set/reset validation variables
        valid_loss = 0
        accuracy = 0
        
        valid_step = 0
        
        ## set model to eval mode and deactivate gradients for validation pass
        model.eval()
        
        print(f'\tBeginning validation for epoch {e+1}')
        
        with torch.no_grad():
            for images, labels in validloader:
                
                valid_step += 1
                if valid_step == 1 or valid_step % 10 == 0:
                    print(f'\t\tValidation step {valid_step}')
                
                ## send images, labels to device
                images, labels = images.to(device), labels.to(device)
                
                ## run validation step
                valid_logps = model(images)
                valid_batch_loss = criterion(valid_logps, labels)
                valid_loss += valid_batch_loss.item()
                valid_output = torch.exp(valid_logps)
                
                ## determine accuracy
                top_p, top_class = valid_output.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
        ## print training stats for epoch
        print(f'\nEpoch {e+1} of {epochs}\n'
              f'\tTraining Loss: {running_loss/len(trainloader):.3f}\n'
              f'\tValidation Loss: {valid_loss/len(validloader):.3f}\n'
              f'\tValidation Accuracy: {accuracy/len(validloader):.3f}\n')
        
        ## add to training summary
        training_summary += (f'\nEpoch {e+1} of {epochs}\n'
              f'\tTraining Loss: {running_loss/len(trainloader):.3f}\n'
              f'\tValidation Loss: {valid_loss/len(validloader):.3f}\n'
              f'\tValidation Accuracy: {accuracy/len(validloader):.3f}\n')
        
        
        
        ## set model to train mode for next epoch
        model.train()
        
    ## print training summary
    print(training_summary)
        
    return model
        