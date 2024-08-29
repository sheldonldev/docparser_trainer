def print_params(model):
    for name, _ in model.named_parameters():
        print(name)
