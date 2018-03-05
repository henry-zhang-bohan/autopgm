from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import itertools
import networkx as nx


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
                        new_cpd = self.merge_cpd_with_parents(cpd, volume)
                        self.cpds[cpd.variable] = (new_cpd, volume + self.cpds[cpd.variable][1])

        # check for cycles
        g = nx.DiGraph(list(self.edges))
        try:
            nx.find_cycle(g)
            return None
        except nx.exception.NetworkXNoCycle:
            pass

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

    def merge_cpd_with_parents(self, cpd, volume):
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
            parents_order = [common_var] + list(filter(lambda x: x != common_var, base[0].variables[1:]))
            base[0].reorder_parents(parents_order)

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
        new_cpd = TabularCPD(variable=base_cpd.variable, variable_card=base_cpd.variable_card,
                             values=new_prob, evidence=new_vars[1:], evidence_card=new_card[1:])

        return self.smooth_cpd(new_cpd)

    @staticmethod
    def smooth_cpd(cpd):
        prob = cpd.values
        evidence_card = cpd.cardinality[1:]
        evidence_card_indices = list(itertools.product(*list(map(lambda x: list(range(x)), list(evidence_card)))))
        for index in evidence_card_indices:
            value_prob = 0.
            for i in range(cpd.variable_card):
                local_cpd = prob[i]
                value_prob += local_cpd[tuple(index)]

            # invalid cpd
            if value_prob == 0:
                reduce_values = []
                for j in range(1, len(cpd.variables)):
                    reduce_values.append((cpd.variables[j], index[j - 1]))

                # reduce one value a time
                marg_prob = []
                for k in range(len(reduce_values)):
                    reduce_values_copy = reduce_values[:]
                    marg_var = reduce_values_copy.pop(k)
                    cpd_copy = cpd.copy()
                    cpd_copy.reduce(reduce_values_copy)
                    np.nan_to_num(cpd_copy.values, copy=False)
                    cpd_copy.marginalize([marg_var[0]])
                    marg_prob.append(cpd_copy.get_values().reshape((cpd.variable_card,)))

                # average out the marginals
                average_prob = np.average(marg_prob, axis=0)

                # fix invalid cpd
                for i in range(cpd.variable_card):
                    prob[tuple([i] + list(index))] = average_prob[i]

        cpd.normalize()
        return cpd
