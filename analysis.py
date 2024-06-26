import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.ticker import AutoMinorLocator
from svg_processor import process_svg
import numpy as np
import os
import re

def get_model_names_with_filters(path, pos_filters, neg_filters = []):
    file_list = os.listdir(path)

    for substring in pos_filters:
        file_list = [file for file in file_list if substring in file]

    for substring in neg_filters:
        file_list = [file for file in file_list if substring not in file]

    model_names = [os.path.splitext(file)[0] for file in file_list]

    return model_names

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

def simple_graph(results, show=True, path=None, filetype=None):
    def plot_it(style, filename, only_show = False):
        with mplstyle.context(style):
            plt.rcParams['font.family'] = 'monospace'
            plt.rcParams['font.monospace'] = 'Iosevka Fixed'
            plt.figure(figsize=(6,4))
            ax = plt.gca()
        
            def forward(x):
                return -np.log2(1-x)
            
            def inverse(x):
                return 1-2**(-x)
            
            ax.set_yscale('function', functions=(forward, inverse))
            #breakpoint()

            for modelname in sort_alphanumeric(results.keys()):
                data = results[modelname]
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
        #    y_ticks = np.insert(y_ticks, 0, y_min)
        #    y_ticks = np.append(y_ticks, y_max)
        
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

            if only_show:
                plt.show()

            else:
                plt.savefig(filename)

            plt.close()
    
    if path is not None:
        if filetype is not None:
            filenames = [path + "_dark." + filetype, path + "_light." + filetype]
        else:
            filenames = [path + "_dark.png", path + "_light.png"]

        styles = ['gruvbox-dark.mplstyle', 'gruvbox-light.mplstyle']

        # print(filenames)
        for style, filename in zip(styles, filenames):
            plot_it(style, filename)
            if filetype == 'svg':
                process_svg(filename)
    
    if show:
        plot_it('gruvbox-dark.mplstyle', '', only_show=True)


def sort_alphanumeric(strings):
    """Sorts a list of strings alphanumerically, correctly handling
    numeric segments, variable lengths, and prioritizes text parts.
    """

    def convert(text):
        """Converts a string into a nested tuple for accurate sorting."""
        parts = text.split('_')
        result = []
        # print(parts)
        for part in parts:
            subparts = re.split(r'(\d+)', part)
            filtered_subparts = [p for p in subparts if p != '']
            # print(f"subparts: {filtered_subparts}")
            for subpart in filtered_subparts:
                if subpart.isdigit():
                    result.append((1, int(subpart)))
                else:
                    result.append((0, subpart))  # Tuple for char comparison
        # print(result)
        return result

    return sorted(strings, key=convert)

def try_me(pos_filters = ['m_SimpleNet2'], neg_filters = [], show=True, path=None, filetype=None):
    model_names = get_model_names_with_filters('models/results', pos_filters, neg_filters)
    # model_names = sort_alphanumeric(model_names)
    # print(model_names)
    results = load_training_results(*model_names)
    simple_graph(results, show=show, path=path, filetype=filetype)
