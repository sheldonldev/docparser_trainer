import h5py  # type: ignore

data_path = '/home/sheldon/data/resource/datax.hd5'


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print(f"    {key}: {val}")


with h5py.File(data_path) as file:
    file.visititems(print_attrs)
