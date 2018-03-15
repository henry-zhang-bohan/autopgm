import os
import pickle
from autopgm.generator import *
from autopgm.estimator import *
from autopgm.helper import *


class Experiment(object):
    def __init__(self, name, data_path, data_dir, split_cols):
        """
        automate setting up and running experiments
        :param name: name of the experiment
        :param data_path: path to the data file (.csv format)
        :param data_dir: directory to store this experiment
        :param split_cols: a list of lists; each sub-list contains the names of the columns selected for each table
        e.g. [['A', 'B'], ['A', 'C'], ['B', 'D']]
        """

        self.name = name
        self.data_path = data_path
        self.data_dir = data_dir
        self.split_cols = split_cols

        # create directory
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        # split train / test sets
        if not os.path.exists(self.data_dir + self.name + '_train.csv') \
                or not os.path.exists(self.data_dir + self.name + '_test.csv'):
            TrainTestSplitter(self.data_path, self.name, split_path=self.data_dir)

        # combine relevant variables
        self.variables = set()
        for i in range(len(self.split_cols)):
            self.variables.update(split_cols[i])
        self.variables = list(self.variables)

        # split tables
        self.split_tables()

        # train single Bayesian network
        self.model = self.train()

        # train merged Bayesian network
        self.merged_model = self.merge()

        print('---------- EVALUATION ----------\n')

        # print log probability
        self.log_prob()

        # print K2 score
        self.k2_score()

        # save plots
        self.plot_edges(merged=False)
        self.plot_edges()

    def split_tables(self):
        if not os.path.exists(self.data_dir + self.name + '_1.csv'):
            CSVSplitter(self.data_dir + self.name + '_train.csv', self.split_cols, self.name, self.data_dir)

    def train(self):
        if not os.path.exists(self.data_dir + self.name + '.p'):
            model = MultipleBayesianEstimator([self.data_dir + self.name + '_train.csv'],
                                              query_targets=self.variables, query_evidence=self.variables).merged_model
            print('model:', model)
            pickle.dump(model, open(self.data_dir + self.name + '.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            model = pickle.load(open(self.data_dir + self.name + '.p', 'rb'))
        return model

    def merge(self):
        if not os.path.exists(self.data_dir + self.name + '_merged.p'):
            # input files
            file_names = []
            for i in range(len(self.split_cols)):
                file_names.append('{}{}_{}.csv'.format(self.data_dir, self.name, (i + 1)))

            # train merged model
            model = MultipleBayesianEstimator(file_names, query_targets=self.variables,
                                              query_evidence=self.variables).merged_model
            print('merged:', model)
            pickle.dump(model, open(self.data_dir + self.name + '_merged.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            model = pickle.load(open(self.data_dir + self.name + '_merged.p', 'rb'))
        return model

    def log_prob(self):
        log_prob = BayesianLogProbability(self.model, self.data_dir + self.name + '_test.csv')
        print('Single model log probability:', log_prob.calculate_log_prob())

        log_prob = BayesianLogProbability(self.merged_model, self.data_dir + self.name + '_test.csv')
        print('Merged model log probability:', log_prob.calculate_log_prob())

        print('\n')

    def k2_score(self):
        data = pandas.read_csv(self.data_dir + self.name + '_test.csv')
        print('Single model K2 score:', K2Score(data).score(self.model))
        print('Merged model K2 score:', K2Score(data).score(self.merged_model))

        print('\n')

    def plot_edges(self, show=False, merged=True):
        import matplotlib.pyplot as plt
        import networkx as nx

        model = self.merged_model if merged else self.model
        fig_name = self.data_dir + (self.name + '_merged.png' if merged else self.name + '.png')
        edges = [(X[:2].capitalize(), Y[:2].capitalize()) for (X, Y) in model.edges]

        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=750)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)

        if not os.path.exists(fig_name):
            plt.savefig(fig_name)

        if show:
            plt.show()
