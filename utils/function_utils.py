import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def from_debug_name_to_name(name: str):
    if name.endswith('-dbg-plugin'):
        return name.split('-dbg-plugin')[0]
    if name[-7:] == '-dbgsym':
        return name[:-7]
    if name[-8:] == '-dbgsym\n':
        return name[:-8]
    elif name[-4:] == '-dbg':
        return name[:-4]
    return None


def flatten_python_list(list_of_lists):
    total_list = []
    for sublist in list_of_lists:
        total_list += sublist
    return total_list


def check_if_function_is_not_interesting(subtoken: str):
    if subtoken == '<s>':
        return True
    if subtoken == '</s>':
        return True
    return False
