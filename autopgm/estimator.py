from autopgm.external.HillClimbSearch import HillClimbSearch
from autopgm.external.K2Score import K2Score
from autopgm.merger import BayesianMerger, BayesianMergerOld
from autopgm.parser import *
from math import inf
import multiprocessing as mp


class SingleBayesianEstimator(object):
    def __init__(self, single_file_parser, inbound_nodes, outbound_nodes, known_independencies=[],
                 n_random_restarts=0, random_restart_length=0):
        self.single_file_parser = single_file_parser
        self.model = None
        self.hill_climb_search = HillClimbSearch(self.single_file_parser.data_frame,
                                                 inbound_nodes=inbound_nodes,
                                                 outbound_nodes=outbound_nodes,
                                                 known_independencies=known_independencies,
                                                 n_random_restarts=n_random_restarts,
                                                 random_restart_length=random_restart_length)

    def random_restart(self, start=None):
        self.model = self.hill_climb_search.random_restart(tabu_length=3, start=start)
        return self.model.edges

    def fit(self):
        self.model.fit(self.single_file_parser.data_frame)
        return self.model


class MultipleBayesianEstimatorOld(object):
    def __init__(self, file_names, query_targets=[], query_evidence=[], inbound_nodes=[], outbound_nodes=[],
                 known_independencies=[], n_random_restarts=0, random_restart_length=0, start=None):
        self.multiple_file_parser = MultipleFileParser(file_names, query_targets=query_targets,
                                                       query_evidence=query_evidence)
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = outbound_nodes
        self.known_independencies = known_independencies

        self.orientations = self.multiple_file_parser.orientations
        self.merged_model = None
        self.max_score = -inf

        # single file case
        if len(self.orientations) == 0:
            self.orientations = [{}]

        # select the best model from all orientations
        with mp.Pool(processes=mp.cpu_count()) as pool:
            o_models = [pool.apply_async(self.orientation_model,
                                         args=(orientation, start, n_random_restarts, random_restart_length)) for
                        orientation in self.orientations]
            orientation_models = [o.get() for o in o_models]

        self.merged_model, self.max_score = max(orientation_models, key=lambda x: x[1])

    def orientation_model(self, orientation, start=None, n_random_restarts=0, random_restart_length=0):
        inbound_nodes = []
        outbound_nodes = []
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            inbound_nodes.append(self.inbound_nodes[:])
            outbound_nodes.append(self.outbound_nodes[:])
        for variable in orientation.keys():
            for position in orientation[variable].keys():
                if orientation[variable][position] == 0:
                    outbound_nodes[position].append(variable)

        # single parsers
        current_models = []
        current_data_volumes = []
        total_score = 0.
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            parser = self.multiple_file_parser.single_file_parsers[i]
            estimator = SingleBayesianEstimator(parser, inbound_nodes[i], outbound_nodes[i],
                                                known_independencies=self.known_independencies[:],
                                                n_random_restarts=n_random_restarts,
                                                random_restart_length=random_restart_length)
            estimator.random_restart(start=start)
            current_model = estimator.fit()
            total_score += K2Score(parser.data_frame).score(current_model)
            current_models.append(current_model)
            current_data_volumes.append(parser.data_frame.shape[0])

        try:
            merged_model = BayesianMergerOld(current_models, current_data_volumes).merge()
        except ValueError:
            merged_model = None
            total_score = -inf

        return merged_model, total_score

    def get_model(self):
        return self.merged_model

    def print_edges(self):
        print(self.merged_model.edges)

    def print_cpds(self):
        for cpd in self.merged_model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)

    def plot_edges(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        edges = [(X[:2].capitalize(), Y[:2].capitalize()) for (X, Y) in self.merged_model.edges]

        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=750)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)

        plt.show()


class MultipleBayesianEstimator(object):
    def __init__(self, file_names, query_targets=[], query_evidence=[], inbound_nodes=[], outbound_nodes=[],
                 known_independencies=[], n_random_restarts=0, random_restart_length=0, start=None):
        self.multiple_file_parser = MultipleFileParser(file_names, query_targets=query_targets,
                                                       query_evidence=query_evidence)
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = outbound_nodes
        self.known_independencies = known_independencies
        self.merged_model = None

        # select the best model from all orientations
        with mp.Pool(processes=mp.cpu_count()) as pool:
            single_models = [pool.apply_async(self.single_model,
                                              args=(parser, n_random_restarts, random_restart_length)) for
                             parser in self.multiple_file_parser.single_file_parsers]
            self.single_models = [sm.get() for sm in single_models]

        # merge models
        data_volumes = list(map(lambda x: x.data_frame.shape[0], self.multiple_file_parser.single_file_parsers))
        self.merged_model = BayesianMerger(self.single_models, data_volumes).merge()

    def single_model(self, parser, n_random_restarts=0, random_restart_length=0):
        estimator = SingleBayesianEstimator(parser, self.inbound_nodes, self.outbound_nodes,
                                            known_independencies=self.known_independencies[:],
                                            n_random_restarts=n_random_restarts,
                                            random_restart_length=random_restart_length)
        estimator.random_restart()
        return estimator.fit()

    def get_model(self):
        return self.merged_model

    def print_edges(self):
        print(self.merged_model.edges)

    def print_cpds(self):
        for cpd in self.merged_model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)

    def plot_edges(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        edges = [(X[:2].capitalize(), Y[:2].capitalize()) for (X, Y) in self.merged_model.edges]

        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=750)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)

        plt.show()
