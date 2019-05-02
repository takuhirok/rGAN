import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import matplotlib.pyplot as plt
import numpy as np


class Logger():
    def __init__(self, out, name='loss', xlabel='iteration'):
        self.out = out
        self.name = name
        self.xlabel = xlabel
        self.txt_file = os.path.join(out, name + '.txt')
        self.plot_file = os.path.join(out, name + '.png')
        self.pkl_file = os.path.join(out, name + '.pkl')

    def log(self, iteration, states, t=None):
        self._print(iteration, states, t)
        self._plot(iteration, states)
        self._pickle(iteration, states)

    def _print(self, iteration, states, t=None):
        if t is not None:
            message = '(iters: %d, time: %.5f) ' % (iteration, t)
        else:
            message = '(iters: %d) ' % (iteration)
        for k, v in states.items():
            message += '%s: %.5f ' % (k, v)

        print(message)
        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)

    def _plot(self, iteration, states):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(states.keys())}
        self.plot_data['X'].append(iteration)
        self.plot_data['Y'].append(
            [states[k] for k in self.plot_data['legend']])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        for i, k in enumerate(self.plot_data['legend']):
            ax.plot(np.array(self.plot_data['X']),
                    np.array(self.plot_data['Y'])[:, i],
                    label=k)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.name)
        l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(self.plot_file,
                    bbox_extra_artists=(l, ),
                    bbox_inches='tight')
        plt.close()

    def _pickle(self, iteration, states):
        if not hasattr(self, 'pickle_data'):
            self.pickle_data = {'iteration': [], 'states': []}
        self.pickle_data['iteration'].append(iteration)
        self.pickle_data['states'].append(states)
        with open(self.pkl_file, 'wb') as f:
            pickle.dump(self.pickle_data, f)
