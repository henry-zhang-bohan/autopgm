import os
import pickle
from autopgm.generator import *
from autopgm.estimator import *
from autopgm.helper import *


class Experiment(object):
    def __init__(self, name, data_path, data_dir, split_cols, synthetic=False):
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
        self.synthetic = synthetic

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

        # train single Bayesian network
        self.model = self.train()

        # synthetic data
        if synthetic:
            self.synthesize_data()

        # split tables
        self.split_tables()

        # train merged Bayesian network
        self.merged_model = self.merge()

        # print log probability
        self.log_prob()

        # print K2 score
        self.k2_score()

        # save plots
        self.plot_edges(merged=False)
        self.plot_edges()

    def split_tables(self):
        if not os.path.exists(self.data_dir + self.name + '_1.csv'):
            file_path = self.data_dir + self.name + ('_train_syn.csv' if self.synthetic else '_train.csv')
            CSVSplitter(file_path, self.split_cols, self.name, self.data_dir)

    def train(self):
        if not os.path.exists(self.data_dir + self.name + '.p'):
            model = MultipleBayesianEstimator([self.data_dir + self.name + '_train.csv'], n_random_restarts=0,
                                              query_targets=self.variables, query_evidence=self.variables).merged_model
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

    def synthesize_data(self):
        if not os.path.exists(self.data_dir + self.name + '_train_syn.csv'):
            CSVWriter(self.model, self.data_dir + self.name + '_train_syn.csv', size=1000000)

    def log_prob(self):
        print('----- log likelihood of the test set -----')

        joint_distribution = JointDistribution(self.data_dir + self.name + '_train.csv', self.variables)
        log_prob = JointLogProbability(self.data_dir + self.name + '_test.csv', joint_distribution, self.variables)
        print('Training set tabular joint distribution:\t', log_prob.calculate_log_prob())

        joint_distribution = JointDistribution(self.data_dir + self.name + '_test.csv', self.variables)
        log_prob = JointLogProbability(self.data_dir + self.name + '_test.csv', joint_distribution, self.variables)
        print('Test set tabular joint distribution:\t\t', log_prob.calculate_log_prob())

        log_prob = BayesianLogProbability(self.model, self.data_dir + self.name + '_test.csv')
        print('Independent Bayesian network:\t\t\t\t', log_prob.calculate_log_prob())

        log_prob = BayesianLogProbability(self.merged_model, self.data_dir + self.name + '_test.csv')
        print('Merged Bayesian network:\t\t\t\t\t', log_prob.calculate_log_prob())

        print('\n')

    def k2_score(self):
        print('----- K2 score -----')

        data = pandas.read_csv(self.data_dir + self.name + '_test.csv')
        print('Independent BN:\t', K2Score(data).score(self.model))
        print('Merged BN:\t\t', K2Score(data).score(self.merged_model))

        print('\n')

    def plot_edges(self, show=False, merged=True):
        import matplotlib.pyplot as plt
        import networkx as nx

        model = self.merged_model if merged else self.model
        plt.figure()
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

    def print_model_cpds(self):
        for cpd in self.model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)

    def print_merged_model_cpds(self):
        for cpd in self.merged_model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)