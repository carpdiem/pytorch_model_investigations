import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os

def get_model_names_with_prefix(path, prefix):
    file_list = os.listdir(path)

    filtered_files = [file for file in file_list if file.startswith(prefix)]

    model_names = [os.path.splitext(file)[0] for file in filtered_files]

    return model_names

def load_training_results(*models, path='models/results'):
    results = {}
    for modelname in models:
        csv_path = os.path.join(path, modelname, modelname + '.csv')

        data = np.loadtxt(csv_path, delimiter=',', skiprows=1, dtype=float)
        headers = np.loadtxt(csv_path, delimiter=',', max_rows=1, dtype=str)

        results[modelname] = {header: values for header, values in zip(headers, data.T)}
    
    return results

def simple_graph(results, show=True, path=None):
    plt.figure(figsize=(6,4))
    ax = plt.gca()

    def forward(x):
        return -np.log2(1-x)
    
    def inverse(x):
        return 1-2**(-x)
    
    ax.set_yscale('function', functions=(forward, inverse))

    for modelname, data in results.items():
        ax.plot(data['FLOPs'], data['Accuracy'], label = modelname)
    
    plt.legend()
    plt.xlabel('FLOPs')
    plt.ylabel('Accuracy')
    plt.title('FLOPs vs Accuracy')


    # Do some trickery to make the "funny log" 1-2^x scale work
    y_min, y_max = ax.get_ylim()
    
    high_tick_exponent = np.ceil(np.log2(1 - y_max))
    low_tick_exponent = np.floor(np.log2(1 - y_min))

    y_ticks = np.arange(low_tick_exponent, high_tick_exponent - 1, -1)
    y_ticks = 1. - 2**y_ticks
    y_ticks = np.insert(y_ticks, 0, y_min)
    y_ticks = np.append(y_ticks, y_max)

    ax.set_yticks(y_ticks)
    ax.tick_params(which='major', length=4)

    # Set some minor ticks for the "funny log" scale to make it clear what's going on
    minor_ticks = []
    num_minor_ticks = 8
    for i in range(len(y_ticks) - 1):
        start = y_ticks[i]
        end = y_ticks[i+1]
        minor_ticks.extend(np.linspace(start, end, num_minor_ticks + 1)[1:])
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(which='minor', length=2, color='gray')

    plt.tight_layout()
    
    if path is not None:
        plt.savefig(path)
    
    if show:
        plt.show()


def try_me(model_prefix = 'm_SimpleNet2'):
    model_names = get_model_names_with_prefix('models/results', model_prefix)
    results = load_training_results(*model_names)
    simple_graph(results)