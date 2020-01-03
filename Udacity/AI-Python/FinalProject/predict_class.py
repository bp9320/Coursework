import torch

def predict(image, model, idx_to_class, topk):
    ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
        Inputs:
            image: (tensor) prepared image
            model: trained image classification model
            idx_to_class: (dict) link between index and appropriate class
            topk: (int) number of top classes to display
        Returns: top k probabilities and predictions
    
    '''
    
    print('*** Starting Prediction ***')
    
    ## Prepare the image with proper dimensions
    image = image.unsqueeze_(0)
    
    ## Get predictions
    model.eval()
    
    log_ps = model(image)
    output = torch.exp(log_ps)
        
    top_ps, top_classes = output.topk(topk, dim=1)
    
    ## convert indexes to classes
    classes = [idx_to_class[int(idx)] for idx in top_classes[0]]

    
    
    ## convert predictions from tensor to list
    pred = top_ps.squeeze().tolist()
    
    print('*** Prediction Complete ***')
    
    return (pred, classes)