from external.HillClimbSearch import HillClimbSearch
from external.K2Score import K2Score
from merger import BayesianMerger
from math import inf


class SingleBayesianEstimator(object):
    def __init__(self, single_file_parser, inbound_nodes, outbound_nodes):
        self.single_file_parser = single_file_parser
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = outbound_nodes
        self.model = None
        self.hill_climb_search = HillClimbSearch(self.single_file_parser.data_frame,
                                                 inbound_nodes=inbound_nodes,
                                                 outbound_nodes=outbound_nodes)

    def initial_edge_estimate(self):
        self.model = self.hill_climb_search.estimate(tabu_length=3)
        return self.model.edges

    def random_restart(self):
        self.model = self.hill_climb_search.random_restart(tabu_length=3)
        return self.model.edges

    def fit(self):
        self.model.fit(self.single_file_parser.data_frame)
        return self.model


class MultipleBayesianEstimator(object):
    def __init__(self, multiple_file_parser):
        self.orientations = multiple_file_parser.orientations
        self.merged_model = None
        self.max_score = -inf

        for orientation in self.orientations:
            inbound_nodes = []
            outbound_nodes = []
            for i in range(len(multiple_file_parser.single_file_parsers)):
                inbound_nodes.append([])
                outbound_nodes.append([])
            for variable in orientation.keys():
                for position in orientation[variable].keys():
                    if orientation[variable][position] == 0:
                        outbound_nodes[position].append(variable)
                    else:
                        inbound_nodes[position].append(variable)

            # single parsers
            current_models = []
            total_score = 0.
            for i in range(len(multiple_file_parser.single_file_parsers)):
                parser = multiple_file_parser.single_file_parsers[i]
                estimator = SingleBayesianEstimator(parser, inbound_nodes[i], outbound_nodes[i])
                #estimator.initial_edge_estimate()
                estimator.random_restart()
                current_model = estimator.fit()
                total_score += K2Score(parser.data_frame).score(current_model)
                current_models.append(current_model)

            if total_score > self.max_score:
                try:
                    self.merged_model = BayesianMerger(current_models).merge()
                    self.max_score = max(total_score, self.max_score)
                except ValueError:
                    continue
