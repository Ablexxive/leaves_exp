from sacred import Experiment

ex = Experiment('config_test')

@ex.config
def cfg():
    C = 10.0
    gamma = 0.7
    params = {
        'C':C,
        'gamma':gamma,
        'learning_rate':10,
        'file_dir':"testing.csv",
    }

@ex.named_config
def variant1():
    C=100000
    params = {}

@ex.automain
def run(params):
    print(type(params['C']))
    print(params)
    return params['C']
