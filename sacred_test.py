from sacred import Experiment

ex = Experiment('config_test')

@ex.config
def cfg():
    params = {
        'C':1.0,
        'gamma':0.7,
        'learning_rate':10,
        'file_dir':"testing.csv",
    }

@ex.automain
def run(params):
    print(params)
    return params['C']
