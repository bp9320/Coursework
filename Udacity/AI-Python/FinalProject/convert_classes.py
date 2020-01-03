import json

def convert_classes(classes, category_path):
    """
        Converts the indexes from numerical strings to class names
        Inputs:
            classes: (list of strings) list of the top k classes from prediction function
            category_path: (str) path to category names .json file
        Returns:
            class_names: list of classes converted to names
    """
    
    print('*** Start Converting Classes To Names ***')
    
    ## Code provided by Udacity in final project workbook
    with open(category_path, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name[i] for i in classes]
    
    print('*** Converting Classes To Names Completed ***')
    
    return class_names

