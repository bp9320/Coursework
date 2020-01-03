def print_predictions(probs, class_names):
    """
        Print the predictions in nice format
        Inputs:
            probs: list of probabilities from prediction
            class_names: list of class names from prediction
        Returns:
            Nothing. Prints results to screen.
    """
    
    print('\n\n*** Results ***\n')
    
    for prob, name in zip(probs, class_names):
        print(f'Class Name: {name.title()}   Probability: {prob:.3f}')