from autopgm.external.HillClimbSearch import HillClimbSearch, GlobalHillClimbSearch
from autopgm.external.K2Score import K2Score
from autopgm.merger import BayesianMerger
from autopgm.parser import *
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from math import inf
import multiprocessing as mp
import random


class SingleBayesianEstimator(object):
    def __init__(self, single_file_parser, inbound_nodes, outbound_nodes, known_independencies=[],
                 n_random_restarts=0, random_restart_length=0, scores=None, index=None, lr_variables=[]):
        self.single_file_parser = single_file_parser
        self.model = None
        self.hill_climb_search = HillClimbSearch(self.single_file_parser.data_frame, scores=scores, index=index,
                                                 inbound_nodes=inbound_nodes,
                                                 outbound_nodes=outbound_nodes,
                                                 known_independencies=known_independencies,
                                                 n_random_restarts=n_random_restarts,
                                                 random_restart_length=random_restart_length,
                                                 lr_variables=lr_variables)

    def random_restart(self, start=None):
        self.model = self.hill_climb_search.random_restart(tabu_length=3, start=start)
        return self.model.edges

    def fit(self):
        self.model.cpds = []
        self.model.fit(self.single_file_parser.data_frame,
                       estimator=BayesianEstimator,
                       prior_type='BDeu',
                       equivalent_sample_size=1)
        return self.model


class MultipleBayesianEstimator(object):
    def __init__(self, file_names, query_targets=[], query_evidence=[], inbound_nodes=[], outbound_nodes=[],
                 known_independencies=[], n_random_restarts=0, random_restart_length=0, start=None,
                 max_orientations=None, random_state=None, lr_variables=[]):
        self.multiple_file_parser = MultipleFileParser(file_names, query_targets=query_targets,
                                                       query_evidence=query_evidence)
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = outbound_nodes
        self.known_independencies = known_independencies
        self.lr_variables = lr_variables

        self.orientations = self.multiple_file_parser.orientations
        self.merged_model = None
        self.max_score = -inf
        self.lr_learnable = []

        # single file case
        if len(self.orientations) == 0:
            self.orientations = [{}]

        # subsample orientations
        random.shuffle(self.orientations)
        if random_state:
            random.seed(random_state)
        if max_orientations:
            self.orientations = random.sample(self.orientations, max_orientations)

        # cached score dictionary
        scores = {}
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            scores[i] = {}

        # select the best model from all orientations, parallel processing
        with mp.Pool(processes=min(mp.cpu_count(), len(self.orientations))) as pool:
            o_models = [pool.apply_async(self.orientation_model,
                                         args=(orientation, start, n_random_restarts, random_restart_length, scores))
                        for orientation in self.orientations]
            orientation_models = [o.get() for o in o_models]

        # select the model with the highest combined score
        self.merged_model, self.max_score, _ = max(orientation_models, key=lambda x: x[1])
        self.lr_learnable = [model[2] for model in orientation_models]

    def orientation_model(self, orientation, start=None, n_random_restarts=0, random_restart_length=0, scores=None):
        # restrict specified nodes to only have inward / outward edges
        inbound_nodes = []
        outbound_nodes = []
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            inbound_nodes.append(self.inbound_nodes[:])
            outbound_nodes.append(self.outbound_nodes[:])
        for variable in orientation.keys():
            for position in orientation[variable].keys():
                if orientation[variable][position] == 0:
                    outbound_nodes[position].append(variable)

        # learn each individual model
        current_models = []
        current_data_volumes = []
        total_score = 0.
        lr_learnable = []
        for i in range(len(self.multiple_file_parser.single_file_parsers)):
            parser = self.multiple_file_parser.single_file_parsers[i]
            estimator = SingleBayesianEstimator(parser, inbound_nodes[i], outbound_nodes[i], index=i, scores=scores,
                                                known_independencies=self.known_independencies[:],
                                                n_random_restarts=n_random_restarts,
                                                random_restart_length=random_restart_length,
                                                lr_variables=self.lr_variables)
            estimator.random_restart(start=start)
            current_model = estimator.fit()
            total_score += K2Score(parser.data_frame).score(current_model)
            current_models.append(current_model)
            current_data_volumes.append(parser.data_frame.shape[0])
            lr_learnable.append(estimator.hill_climb_search.lr_learnable)

        # merge individual models
        try:
            merged_model = BayesianMerger(current_models, current_data_volumes).merge()
        except ValueError:
            merged_model = None
            total_score = -inf

        return merged_model, total_score, lr_learnable

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

        plt.figure()
        edges = [(X[:2].capitalize(), Y[:2].capitalize()) for (X, Y) in self.merged_model.edges]
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
