import torch


def is_bfloat16_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        if torch.cuda.get_device_properties(device).major >= 8:
            return True
        else:
            return False
    else:
        raise Exception("CUDA not available!")


if __name__ == '__main__':
    print(is_bfloat16_available())

    device = torch.device("cuda")
    for i in range(20):
        print('>>>')
        x = 2**i - 7.6
        print(torch.tensor(x, dtype=torch.float32).to(device))
        print(torch.tensor(x, dtype=torch.float16).to(device))
        print(torch.tensor(x, dtype=torch.bfloat16).to(device))
