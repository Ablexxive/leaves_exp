from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import SlackObserver
ex = Experiment('config_test')

ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='sacred'))
ex.observers.append(SlackObserver.from_config('sacred_slack.json'))

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
def test(params):
    print(type(params['C']))
    print(params)
    return params['C']
