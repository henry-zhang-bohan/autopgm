from autopgm.external.HillClimbSearch import HillClimbSearch, GlobalHillClimbSearch
from autopgm.parser import *
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import pandas


class SimpleBayesianEstimator(object):
    def __init__(self, file_name, n_random_restarts=0, random_restart_length=0, lr_variables=[]):
        data = pandas.read_csv(file_name)
        search = HillClimbSearch(data,
                                 n_random_restarts=n_random_restarts,
                                 random_restart_length=random_restart_length,
                                 lr_variables=lr_variables,
                                 scores=[{}])
        self.model = search.random_restart(tabu_length=3)
        self.model.fit(data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=1)
        self.lr_learnable = search.lr_learnable

    def get_model(self):
        return self.model

    def print_edges(self):
        print(self.model.edges)

    def print_cpds(self):
        for cpd in self.model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)

    def plot_edges(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure()
        edges = [(X[:2].capitalize(), Y[:2].capitalize()) for (X, Y) in self.model.edges]
        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=750)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)

        plt.show()


class GlobalBayesianEstimator(object):
    def __init__(self, file_names, query_targets=[], query_evidence=[]):
        self.file_names = file_names
        self.query_targets = query_targets
        self.query_evidence = query_evidence
        self.multiple_file_parser = MultipleFileParser(file_names, query_targets=query_targets,
                                                       query_evidence=query_evidence, orientations=False)

        # hill climb search using multiple data sources
        self.hill_climb_search = GlobalHillClimbSearch(self.multiple_file_parser)
        self.merged_model = None

        # for parameter learning
        self.model = None
        self.models = []
        self.volumes = []
        self.edges = set()
        self.cpds = {}

        # learn Bayesian Network
        self.learn_structure()
        self.learn_parameter()

        # merge cpds
        self.merged_model = self.merge()

        # record structure history
        self.structure_history = self.hill_climb_search.structure_history

    def learn_structure(self):
        self.merged_model = self.hill_climb_search.random_restart()

    def learn_parameter(self):
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            parser = self.multiple_file_parser.single_file_parsers[i]
            edges = set()
            for (X, Y) in list(self.merged_model.edges):
                if X in parser.variables and Y in parser.variables:
                    edges.add((X, Y))
            model = BayesianModel(list(edges))
            model.fit(parser.data_frame)
            self.models.append(model)
            self.volumes.append(parser.data_frame.shape[0])

    def merge(self):
        for i in range(len(self.models)):
            model = self.models[i]
            volume = self.volumes[i]

            # collect all edges
            self.edges.update(model.edges)

            # cpd's
            for cpd in model.get_cpds():
                if cpd.variable not in self.cpds.keys():
                    self.cpds[cpd.variable] = (cpd, volume)
                else:
                    # replace priors (e.g. P(A)) with conditional probabilities (e.g. P(A|B))
                    if len(cpd.variables) > len(self.cpds[cpd.variable][0].variables):
                        self.cpds[cpd.variable] = (cpd, volume)

                    # merge conditional probabilities using average
                    elif len(cpd.variables) == len(self.cpds[cpd.variable][0].variables) and len(cpd.variables) == 1:
                        old_cpd, old_volume = self.cpds[cpd.variable]
                        new_cpd = []
                        for j in range(len(old_cpd.values)):
                            old_prob = old_cpd.values[j]
                            prob = cpd.values[j]
                            new_cpd.append((old_prob * old_volume + prob * volume) / float(old_volume + volume))
                        new_tabular_cpd = TabularCPD(variable=cpd.variable, variable_card=cpd.variable_card,
                                                     values=[new_cpd], state_names=old_cpd.state_names)
                        self.cpds[cpd.variable] = (new_tabular_cpd, old_volume + volume)

        # new model
        self.model = BayesianModel(self.edges)
        for cpd, volume in self.cpds.values():
            try:
                self.model.add_cpds(cpd)
            except ValueError:
                self.model.add_node(cpd.variable)
                self.model.add_cpds(cpd)

        if self.model.check_model():
            return self.model
        else:
            return None

    def get_model(self):
        return self.merged_model
