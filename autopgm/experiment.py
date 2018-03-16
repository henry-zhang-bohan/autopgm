import os
import pickle
from functools import reduce
from autopgm.generator import *
from autopgm.estimator import *
from autopgm.helper import *
from pgmpy.inference import VariableElimination


class Experiment(object):
    def __init__(self, name, data_path, data_dir, split_cols, synthetic=False, show_scores=True):
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
        self.inference = VariableElimination(self.model)

        # synthetic data
        if synthetic:
            self.synthesize_data()

        # split tables
        self.split_tables()

        # train merged Bayesian network
        self.merged_model = self.merge()
        self.merged_inference = VariableElimination(self.merged_model)

        # state names
        self.model_states, self.merged_model_states = self.state_names()

        if show_scores:
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

    def state_names(self):
        model_state_names = {}
        for cpd in self.model.get_cpds():
            for state in cpd.state_names.keys():
                if state not in model_state_names.keys():
                    model_state_names[state] = cpd.state_names[state]

        merged_model_state_names = {}
        for cpd in self.merged_model.get_cpds():
            for state in cpd.state_names.keys():
                if state not in merged_model_state_names.keys():
                    merged_model_state_names[state] = cpd.state_names[state]

        return model_state_names, merged_model_state_names

    def translate_evidence(self, evidence, merged=False):
        state_names = self.merged_model_states if merged else self.model_states
        new_evidence = {}
        for e in evidence.keys():
            new_evidence[e] = state_names[e].index(evidence[e])
        return new_evidence

    def query(self, variable, evidence, show=False, df=None):
        # load test data
        if df is None:
            df = pandas.read_csv(self.data_dir + self.name + '_test.csv')
        df_q_str = ' & '.join(['{} == {}'.format(var, val) for var, val in evidence.items()])

        # ground truth counting
        if len(df_q_str) > 0:
            df_q = df.query(df_q_str)[variable].value_counts(normalize=True, sort=False).sort_index()
        else:
            df_q = df[variable].value_counts(normalize=True, sort=False).sort_index()

        # prepare queries
        q = self.inference.query([variable], evidence=self.translate_evidence(evidence))[variable]
        merged_q = self.merged_inference.query([variable], evidence=self.translate_evidence(evidence, True))[variable]

        # description
        evidence_str = ', '.join(['{}={}'.format(var, val) for var, val in evidence.items()])
        if len(evidence) > 0:
            prob_str = '\nP({} | {})'.format(variable, evidence_str)
        else:
            prob_str = '\nP({})'.format(variable)

        if show:
            # print probability distribution
            print(prob_str)

            # ground truth
            print('\n(1) Test set distribution:')
            print(df_q)

            # independent model
            print('\n(2) Independent Bayesian network:')
            print(q)

            # merged model
            print('\n(3) Merged Bayesian network:')
            print(merged_q)

        # L1
        try:
            l1 = np.linalg.norm(df_q.values - merged_q.values, 1)
        except ValueError:
            l1 = float('inf')

        # L2
        try:
            l2 = np.linalg.norm(df_q.values - merged_q.values, 2)
        except ValueError:
            l2 = float('inf')

        return {
            'data_frame_query': df_q,
            'inference_query': q,
            'merged_inference_query': merged_q,
            'description': prob_str,
            'l1': l1,
            'l2': l2
        }

    def write_query_to_file(self, query, index):
        with open(self.data_dir + 'queries/' + str(index + 1) + '.txt', 'w') as f:
            print(query['description'].strip(), file=f)

            # ground truth
            print('\n(1) Test set distribution:', file=f)
            print(query['data_frame_query'], file=f)

            # independent model
            print('\n(2) Independent Bayesian network:', file=f)
            print(query['inference_query'], file=f)

            # merged model
            print('\n(3) Merged Bayesian network:', file=f)
            print(query['merged_inference_query'], file=f)

    def all_queries(self):
        import itertools

        variables_intersection = reduce(set.intersection, list(map(lambda x: set(x), self.split_cols)))
        variables_union = set().union(*self.split_cols)
        variables_non_shared = variables_union - variables_intersection
        split_cols = list(map(lambda x: set(x), self.split_cols))
        df_train = pandas.read_csv(self.data_dir + self.name + '_train.csv')
        df_test = pandas.read_csv(self.data_dir + self.name + '_test.csv')

        queries = []
        var_states = {}

        # get all non-shared variables
        for var in variables_non_shared:
            alien_evidence = []
            for alien_var in variables_non_shared:
                if alien_var != var and len(list(filter(lambda x: var in x and alien_var in x, split_cols))) == 0:
                    alien_evidence.append(alien_var)

            # all combinations of evidence
            combinations = []
            for i in range(len(alien_evidence) + 1):
                combo = itertools.combinations(alien_evidence, i)
                combinations.extend(combo)

            evidence_combinations = []
            # all values of combinations
            for combo in combinations:
                evidence_combo = []
                for e in combo:
                    if e in var_states.keys():
                        e_values = var_states[e]
                    else:
                        e_values = sorted(list(df_train[e].unique()))
                        var_states[e] = e_values
                    evidence_combo.append(list(map(lambda x: (e, x), e_values)))
                evidence_combo = list(itertools.product(*evidence_combo))
                evidence_combinations.extend(evidence_combo)

            # convert to dictionary
            for combo in evidence_combinations:
                query = {'variable': var, 'evidence': {}}
                for e in combo:
                    query['evidence'][e[0]] = e[1]
                queries.append(query)

        # compute queries
        computed_queries = []
        for query in queries:
            query_result = self.query(query['variable'], query['evidence'], df=df_test)
            computed_queries.append(query_result)
        computed_queries.sort(key=lambda x: x['l2'])

        if not os.path.exists(self.data_dir + 'queries/'):
            os.makedirs(self.data_dir + 'queries/')
        for i in range(len(computed_queries)):
            self.write_query_to_file(computed_queries[i], i)
