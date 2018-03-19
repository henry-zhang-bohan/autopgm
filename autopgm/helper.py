import pandas
from numpy import product
from math import log, exp


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


class JointLogProbability(object):
    def __init__(self, data_file_name, distribution, variables, equivalent_sample_size=1.):
        self.distribution = distribution
        self.data_frame = pandas.read_csv(data_file_name)[variables].dropna()
        self.equivalent_sample_size = equivalent_sample_size

    def calculate_log_prob(self):
        log_prob = 0.
        # normalizing zero probabilities, normalizing factor
        normalizing_factor = float(self.distribution.data_size) / float(
            self.distribution.data_size + self.equivalent_sample_size)
        # BDeu prior
        smooth_prob = (self.equivalent_sample_size * normalizing_factor) / (
                float(self.distribution.data_size) * float(self.distribution.combination_count))
        # calculate log probability for every data point
        for i, row in self.data_frame.iterrows():
            values = tuple(row.values.tolist())
            if values in self.distribution.distribution.keys():
                prob = self.distribution.distribution[values] * normalizing_factor
            else:
                prob = smooth_prob
            log_prob += log(prob)

        return log_prob


class JointDistribution(object):
    def __init__(self, data_file_name, variables):
        self.data_frame = pandas.read_csv(data_file_name)[variables].dropna()
        self.domains = self.variable_domains()
        self.data_size = self.data_frame.shape[0]
        self.combination_count = product(list(map(lambda x: len(x), self.domains)))
        self.distribution = self.calculate_probability()

    def variable_domains(self):
        domains = []
        for variable in self.data_frame.columns:
            domains.append(list(self.data_frame[variable].unique()))
        return domains

    def calculate_probability(self):
        distribution = {}
        for i, row in self.data_frame.iterrows():
            values = tuple(row.values.tolist())
            if values in distribution.keys():
                distribution[values] += 1. / self.data_size
            else:
                distribution[values] = 1. / self.data_size
        return distribution


class KLDivergence(object):
    def __init__(self, model, data_file_name, variables):
        self.model = model
        self.variables = variables
        self.distribution = JointDistribution(data_file_name, variables)
        self.joint_probabilities = self.distribution.calculate_probability()
        self.cpd_dict = {}
        for cpd in self.model.cpds:
            self.cpd_dict[cpd.variable] = cpd

    def calculate_kl_divergence(self):
        d = 0.
        for values in self.joint_probabilities.keys():
            log_prob = 0.
            # value assignments
            assignments = {}
            for i in range(len(self.variables)):
                assignments[self.variables[i]] = values[i]

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

            # calculate divergence
            joint_prob = self.joint_probabilities[values]
            model_prob = exp(log_prob)
            d += joint_prob * (log(model_prob) - log(joint_prob))
        return -d
