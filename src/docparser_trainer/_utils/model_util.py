def print_params(model):
    for name, _ in model.named_parameters():
        print(name)


def print_params_dtype(model):
    for name, param in model.named_parameters():
        print(name, param.dtype)


def print_model_info(model):
    model_size = sum(param.numel() for param in model.parameters())
    print(f"original model size: {model_size:,} params".capitalize())

    param_gpu_usage = model_size * model.dtype.itemsize
    print(f"original parameters gpu usage: {param_gpu_usage:,} bytes".capitalize())

    gradient_gpu_usage = model_size * model.dtype.itemsize
    print(f"original gradient gpu usage: {gradient_gpu_usage:,} bytes".capitalize())

    optimizer_gpu_usage = model_size * model.dtype.itemsize * 2
    print(f"original optimizer gpu usage: {optimizer_gpu_usage:,} bytes".capitalize())

    model_gpu_usage = param_gpu_usage + gradient_gpu_usage + optimizer_gpu_usage
    print(f"original model gpu usage: {model_gpu_usage:,} bytes".capitalize())
