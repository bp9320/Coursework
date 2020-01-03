from get_inputs import get_predict_inputs, check_pth_file, check_cat_file
from data_prep import process_image
from build_model import build_predict_model
from predict_class import predict
from convert_classes import convert_classes
from print_predictions import print_predictions

def main():
    """
    Main function
    """
    
    ## get input arguments
    input_args = get_predict_inputs()
#     print(input_args)
    
    ## check that checkpoint is .pth file
    input_args.checkpoint = check_pth_file(input_args.checkpoint)
#     print(input_args)
    
    ## check that categories file is .json file
    input_args.category_names = check_cat_file(input_args.category_names)
#     print(input_args)
    
    ## prepare image for prediction model
    image = process_image(input_args.img_path)
#     print(type(image), image.shape)

    ## build model from checkpoint
    model, idx_to_class = build_predict_model(input_args.checkpoint)
#     print(model, idx_to_class)

    ## run prediction
    probs, classes = predict(image, model, idx_to_class, input_args.top_k)
#     print(probs, classes)
    
    ## get class names from json
    class_names = convert_classes(classes, input_args.category_names)
#     print(probs, class_names)
    
    ## pretty print results
    print_predictions(probs, class_names)
    
    

if __name__ == '__main__':
    main()