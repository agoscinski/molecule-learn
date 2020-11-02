
def class_name(object):
    cl = object.__class__
    return f'{cl.__module__}.{cl.__name__}'
