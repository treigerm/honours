MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f
    
    return decorator

def get_model(model_name, *args, **kwargs):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](*args, **kwargs)
    else:
        raise ValueError("Model class does not exist {}.".format(model_name))