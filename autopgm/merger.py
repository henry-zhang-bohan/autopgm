from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import itertools


class BayesianMergerOld(object):
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
            try:
                self.model.add_cpds(cpd)
            except ValueError:
                self.model.add_node(cpd.variable)
                self.model.add_cpds(cpd)

        if self.model.check_model():
            return self.model
        else:
            # TODO: raise errors
            return None


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

            # cpd's
            for cpd in model.get_cpds():
                # new cpd
                if cpd.variable not in self.cpds.keys():
                    self.cpds[cpd.variable] = (cpd, volume)
                else:
                    old_cpd_len = len(self.cpds[cpd.variable][0].variables)
                    new_cpd_len = len(cpd.variables)

                    # conditional on other variables now
                    if old_cpd_len == 1 and new_cpd_len > old_cpd_len:
                        self.cpds[cpd.variable] = (cpd, volume)

                    # being the parent now and before
                    elif new_cpd_len == old_cpd_len and new_cpd_len == 1:
                        old_cpd, old_volume = self.cpds[cpd.variable]
                        new_cpd = []
                        for j in range(len(old_cpd.values)):
                            old_prob = old_cpd.values[j]
                            prob = cpd.values[j]
                            new_cpd.append((old_prob * old_volume + prob * volume) / float(old_volume + volume))
                        new_tabular_cpd = TabularCPD(variable=cpd.variable, variable_card=cpd.variable_card,
                                                     values=[new_cpd])
                        self.cpds[cpd.variable] = (new_tabular_cpd, old_volume + volume)

                    # conflicting parents
                    elif old_cpd_len > 1 and new_cpd_len > 1:
                        if volume > self.cpds[cpd.variable][1]:
                            base = cpd, volume
                            top = self.cpds[cpd.variable]
                        else:
                            base = self.cpds[cpd.variable]
                            top = cpd, volume

                        # find shared parents
                        common_vars = list(filter(lambda x: x in top[0].variables[1:], base[0].variables[1:]))
                        for common_var in common_vars:
                            # marginalize w.r.t. the common variable
                            base_marg = base[0].copy()
                            top_marg = top[0].copy()
                            base_marg.marginalize(list(filter(lambda x: x != common_var, base_marg.variables[1:])))
                            top_marg.marginalize(list(filter(lambda x: x != common_var, top_marg.variables[1:])))

                            # merge marginalized probability
                            new_marg = base_marg.copy()
                            new_marg.values = (base_marg.values * float(base[1]) + top_marg.values * float(top[1])) \
                                              / float(base[1] + top[1])
                            new_marg.normalize()

                            # update posterior probability
                            parents_order = [common_var] + list(
                                filter(lambda x: x != common_var, base[0].variables[1:]))
                            base_cpd = base[0]
                            base_cpd.reorder_parents(parents_order)

                            # loop through child range and common_var range
                            for i in range(base[0].cardinality[0]):
                                for j in range(base[0].cardinality[1]):
                                    base[0].values[i, j] *= new_marg.values[i, j] / base_marg.values[i, j]
                            base[0].normalize()

                        # reorder
                        base[0].reorder_parents(common_vars + list(filter(
                            lambda x: x not in common_vars, base[0].variables[1:])))
                        top[0].reorder_parents(common_vars + list(filter(
                            lambda x: x not in common_vars, top[0].variables[1:])))

                        # copy cpd's
                        base_cpd = base[0].copy()
                        top_cpd = top[0].copy()

                        # new cpd
                        new_card = np.concatenate((base_cpd.cardinality, top_cpd.cardinality[len(common_vars) + 1:]))
                        new_card_indices = list(itertools.product(*list(map(lambda x: list(range(x)), list(new_card)))))
                        new_vars = base_cpd.variables + top_cpd.variables[len(common_vars) + 1:]
                        new_prob = np.zeros(new_card)

                        # iterate all indices to construct new cpd
                        for index in new_card_indices:
                            index = list(index)
                            shared_portion = index[:len(common_vars) + 1]
                            base_portion = tuple(index[:len(base_cpd.variables)])
                            top_portion = tuple(shared_portion + index[len(base_cpd.variables):])
                            new_prob[tuple(index)] = base_cpd.values[base_portion] * top_cpd.values[top_portion]

                        # transform cpd
                        new_prob = new_prob.reshape((base_cpd.variable_card, np.product(new_card[1:])))
                        new_prob = (new_prob / new_prob.sum(axis=0))

                        # check for nan columns, and replace with uniform distribution
                        for i in range(new_prob.shape[1]):
                            is_valid = True
                            for j in new_prob[:, i]:
                                if np.isnan(j):
                                    is_valid = False
                                    break
                            if not is_valid:
                                new_prob[:, i] = float(1) / float(new_prob.shape[0])

                        # construct new cpd
                        new_cpd = TabularCPD(variable=base_cpd.variable, variable_card=base_cpd.variable_card,
                                             values=new_prob, evidence=new_vars[1:], evidence_card=new_card[1:])
                        # new_cpd.normalize()
                        self.cpds[cpd.variable] = (new_cpd, volume + self.cpds[cpd.variable][1])

        # new model
        self.model = BayesianModel(self.edges)
        for cpd, volume in self.cpds.values():
            cpd.normalize()
            try:
                self.model.add_cpds(cpd)
            except ValueError:
                self.model.add_node(cpd.variable)
                self.model.add_cpds(cpd)

            if not np.allclose(cpd.to_factor().marginalize([cpd.variable], inplace=False).values.flatten('C'),
                               np.ones(np.product(cpd.cardinality[:0:-1])),
                               atol=0.01):
                print(cpd)

        if self.model.check_model():
            return self.model
        else:
            # TODO: raise errors
            return None
