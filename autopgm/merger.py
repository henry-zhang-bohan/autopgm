from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


class BayesianMerger(object):
    def __init__(self, models, volumes):
        self.models = models
        self.volumes = volumes
        self.edges = set()
        self.cpds = {}
        self.model = None

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
