from HillClimbSearch import HillClimbSearch
from pgmpy.estimators import StructureEstimator, K2Score, BicScore


class SingleBayesianEstimator(object):
    def __init__(self, single_file_parser):
        self.single_file_parser = single_file_parser
        self.hill_climb_search = HillClimbSearch(self.single_file_parser.data_frame,
                                                 outbound_nodes=self.single_file_parser.shared_variables)
        self.model = None

    def initial_edge_estimate(self):
        self.model = self.hill_climb_search.estimate(tabu_length=3)
        return self.model.edges

    def fit(self):
        self.model.fit(self.single_file_parser.data_frame)
        return self.model
