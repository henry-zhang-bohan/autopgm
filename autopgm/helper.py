import pandas
from math import log


class BayesianLogProbability(object):
    def __init__(self, model, data_file_name):
        self.model = model
        self.data_frame = pandas.read_csv(data_file_name)[list(model.nodes)].dropna()
        self.variables = list(self.data_frame.columns)
        self.cpd_dict = {}
        for cpd in self.model.cpds:
            self.cpd_dict[cpd.variable] = cpd

    def calculate_log_prob(self):
        log_prob = 0.
        for i, row in self.data_frame.iterrows():
            assignments = row.to_dict()
            for cpd in self.model.cpds:
                # collect parents' values
                evidence = []
                for e in cpd.variables[1:]:
                    value = cpd.state_names[e].index(assignments[e])
                    evidence.append((e, value))
                cpd_copy = cpd.copy()
                cpd_copy.reduce(evidence)

                # P(var | theta)
                prob = cpd_copy.values[cpd.state_names[cpd.variable].index(assignments[cpd.variable])]
                log_prob += log(prob)

        return log_prob
