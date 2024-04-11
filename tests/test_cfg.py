from docparser_trainer import _cfg


def test_cfg():
    for name in dir(_cfg):
        attr = getattr(_cfg, name)
        if name.isupper() and not callable(attr):
            print(f"{name}: {attr}")
