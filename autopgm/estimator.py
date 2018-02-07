from external.HillClimbSearch import HillClimbSearch
from external.K2Score import K2Score
from merger import BayesianMerger
from math import inf


class SingleBayesianEstimator(object):
    def __init__(self, single_file_parser, inbound_nodes, outbound_nodes,
                 n_random_restarts=10, random_restart_length=5):
        self.single_file_parser = single_file_parser
        self.model = None
        self.hill_climb_search = HillClimbSearch(self.single_file_parser.data_frame,
                                                 inbound_nodes=inbound_nodes,
                                                 outbound_nodes=outbound_nodes,
                                                 n_random_restarts=n_random_restarts,
                                                 random_restart_length=random_restart_length)

    def random_restart(self, start=None):
        self.model = self.hill_climb_search.random_restart(tabu_length=3, start=start)
        return self.model.edges

    def fit(self):
        self.model.fit(self.single_file_parser.data_frame)
        return self.model


class MultipleBayesianEstimator(object):
    def __init__(self, multiple_file_parser, n_random_restarts=10, random_restart_length=5, start=None):
        self.orientations = multiple_file_parser.orientations
        self.merged_model = None
        self.max_score = -inf

        # single file case
        if len(self.orientations) == 0:
            self.orientations = [{}]

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
            current_data_volumes = []
            total_score = 0.
            for i in range(len(multiple_file_parser.single_file_parsers)):
                parser = multiple_file_parser.single_file_parsers[i]
                estimator = SingleBayesianEstimator(parser, inbound_nodes[i], outbound_nodes[i],
                                                    n_random_restarts=n_random_restarts,
                                                    random_restart_length=random_restart_length)
                estimator.random_restart(start=start)
                current_model = estimator.fit()
                total_score += K2Score(parser.data_frame).score(current_model)
                current_models.append(current_model)
                current_data_volumes.append(parser.data_frame.shape[0])

            if total_score > self.max_score:
                try:
                    self.merged_model = BayesianMerger(current_models, current_data_volumes).merge()
                    self.max_score = max(total_score, self.max_score)
                except ValueError:
                    continue

    def print_edges(self):
        print(self.merged_model.edges)

    def print_cpds(self):
        for cpd in self.merged_model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)
