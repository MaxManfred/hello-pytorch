import inspect


def print_example(function_to_execute):
    print('###############################################################')
    print('')
    function_to_execute.__call__()
    print('')


def print_all_examples(module_name):
    for (name, func) in inspect.getmembers(module_name, inspect.isfunction):
        if name.startswith('example_'):
            print_example(func)
