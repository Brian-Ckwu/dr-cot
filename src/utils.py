from argparse import Namespace

def dict_to_namespace(d: dict) -> Namespace:
    """Converts a dictionary to a namespace recursively."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)

# manual testing
if __name__ == "__main__":
    d = {"a": {"c": 2, "d": {"e": 3}}, "b": 3}
    n = dict_to_namespace(d)
    print(n)
    print(n.a.c)
    print(n.a.d.e)
