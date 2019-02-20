DATASET_REGISTRY = {}

def register_dataset(dataset_name):
    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f
    
    return decorator

def get_dataset(dataset_name, *args, **kwargs):
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name](*args, **kwargs)
    else:
        raise ValueError("Dataset class does not exist {}.".format(dataset_name))