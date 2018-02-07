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

            # edges
            self.edges.update(model.edges)
            # cpds
            for cpd in model.get_cpds():
                if cpd.variable not in self.cpds.keys():
                    self.cpds[cpd.variable] = (cpd, volume)
                else:
                    if len(cpd.variables) > len(self.cpds[cpd.variable][0].variables):
                        self.cpds[cpd.variable] = (cpd, volume)
                    elif len(cpd.variables) == len(self.cpds[cpd.variable][0].variables) and len(cpd.variables) == 1:
                        old_cpd, old_volume = self.cpds[cpd.variable]
                        new_cpd = []
                        for j in range(len(old_cpd.values)):
                            old_prob = old_cpd.values[j]
                            prob = cpd.values[j]
                            new_cpd.append((old_prob * old_volume + prob * volume) / float(old_volume + volume))
                        new_tabular_cpd = TabularCPD(variable=cpd.variable, variable_card=cpd.variable_card,
                                                     values=[new_cpd])
                        self.cpds[cpd.variable] = (new_tabular_cpd, old_volume + volume)

        # new model
        self.model = BayesianModel(self.edges)
        for cpd, volume in self.cpds.values():
            self.model.add_cpds(cpd)

        if self.model.check_model():
            return self.model
        else:
            # TODO: raise errors
            return None
