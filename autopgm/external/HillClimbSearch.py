from itertools import permutations
import networkx as nx
from pgmpy.estimators import StructureEstimator
from autopgm.external.K2Score import K2Score
from pgmpy.models import BayesianModel
import random
from collections import defaultdict


class HillClimbSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, inbound_nodes=[], outbound_nodes=[], known_independencies=[],
                 n_random_restarts=10, random_restart_length=5, scores=None, index=0, lr_variables=[], **kwargs):
        """
        Class for heuristic hill climb searches for BayesianModels, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            self.scoring_method = K2Score(data, **kwargs)

        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = outbound_nodes
        self.known_independencies = known_independencies
        self.n_random_restarts = n_random_restarts
        self.random_restart_length = random_restart_length
        self.scores = scores
        self.index = index
        self.lr_variables = lr_variables
        self.lr_learnable = []

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score = self.scoring_method.local_score
        nodes = self.state_names.keys()

        # inbound nodes: outbound edges of prohibited
        prohibited_outbound_edges = set()
        for node in self.inbound_nodes:
            prohibited_outbound_edges.update([(node, X) for X in nodes])

        # outbound nodes: inbound edges of prohibited
        prohibited_inbound_edges = set()
        for node in self.outbound_nodes:
            prohibited_inbound_edges.update([(X, node) for X in nodes])

        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]) -
                               set(self.known_independencies) -
                               prohibited_outbound_edges -
                               prohibited_inbound_edges)

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = list(model.get_parents(Y))
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        # score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                        score_delta = self.get_local_score(Y, new_parents) - self.get_local_score(Y, old_parents)
                        yield (operation, score_delta)

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                # score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                score_delta = self.get_local_score(Y, new_parents) - self.get_local_score(Y, old_parents)
                yield (operation, score_delta)

        for (X, Y) in model.edges():  # (3) flip single edge
            if (Y, X) not in prohibited_inbound_edges and (Y, X) not in prohibited_outbound_edges:
                new_edges = list(model.edges()) + [(Y, X)]
                new_edges.remove((X, Y))
                if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                    operation = ('flip', (X, Y))
                    if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                        old_X_parents = list(model.get_parents(X))
                        old_Y_parents = list(model.get_parents(Y))
                        new_X_parents = old_X_parents + [Y]
                        new_Y_parents = old_Y_parents[:]
                        new_Y_parents.remove(X)
                        if max_indegree is None or len(new_X_parents) <= max_indegree:
                            # score_delta = (local_score(X, new_X_parents) +
                            #                local_score(Y, new_Y_parents) -
                            #                local_score(X, old_X_parents) -
                            #                local_score(Y, old_Y_parents))
                            score_delta = (self.get_local_score(X, new_X_parents) +
                                           self.get_local_score(Y, new_Y_parents) -
                                           self.get_local_score(X, old_X_parents) -
                                           self.get_local_score(Y, old_Y_parents))
                            yield (operation, score_delta)

    def _legal_operations_without_score(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        nodes = self.state_names.keys()

        # inbound nodes: outbound edges of prohibited
        prohibited_outbound_edges = set()
        for node in self.inbound_nodes:
            prohibited_outbound_edges.update([(node, X) for X in nodes])

        # outbound nodes: inbound edges of prohibited
        prohibited_inbound_edges = set()
        for node in self.outbound_nodes:
            prohibited_inbound_edges.update([(X, node) for X in nodes])

        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]) -
                               set(self.known_independencies) -
                               prohibited_outbound_edges -
                               prohibited_inbound_edges)

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = list(model.get_parents(Y))
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        yield operation

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                yield operation

        for (X, Y) in model.edges():  # (3) flip single edge
            if (Y, X) not in prohibited_inbound_edges and (Y, X) not in prohibited_outbound_edges:
                new_edges = list(model.edges()) + [(Y, X)]
                new_edges.remove((X, Y))
                if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                    operation = ('flip', (X, Y))
                    if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                        old_X_parents = list(model.get_parents(X))
                        old_Y_parents = list(model.get_parents(Y))
                        new_X_parents = old_X_parents + [Y]
                        new_Y_parents = old_Y_parents[:]
                        new_Y_parents.remove(X)
                        if max_indegree is None or len(new_X_parents) <= max_indegree:
                            yield operation

    def estimate(self, start=None, tabu_list=[], tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: BayesianModel instance
            The starting point for the local search. By default a completely disconnected network is used.
        tabu_list: list[operations]
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        Returns
        -------
        model: `BayesianModel` instance
            A `BayesianModel` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data, scoring_method=BicScore(data))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """
        epsilon = 1e-8
        nodes = self.state_names.keys()
        if start is None:
            start = BayesianModel()
            start.add_nodes_from(nodes)
        elif not isinstance(start, BayesianModel) or not set(start.nodes()) == set(nodes):
            raise ValueError("'start' should be a BayesianModel with the same variables as the data set, or 'None'.")

        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model, tabu_list, max_indegree):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            print(best_operation)
            print(best_score_delta)

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]

            if len(self.lr_variables) > 0:
                self.lr_learnable.append(self.is_lr_learnable(current_model))

        return current_model

    def random_restart(self, start=None, tabu_length=0, max_indegree=None):
        # starting best model
        if not start:
            best_model = self.estimate(tabu_length=tabu_length, max_indegree=max_indegree)
        else:
            best_model = start
        best_score = K2Score(self.data).score(best_model)

        # iterate random restarts
        for i in range(self.n_random_restarts):
            current_model = best_model.copy()
            n_moves = self.calculate_random_restart_length(i)
            tabu_list = []

            # perform random actions
            for j in range(n_moves):
                operations = []
                for operation in self._legal_operations_without_score(current_model, tabu_list, max_indegree):
                    operations.append(operation)

                try:
                    operation = random.choice(operations)
                except IndexError:
                    continue

                # perform operation
                if operation[0] == '+':
                    current_model.add_edge(*operation[1])
                    tabu_list = ([('-', operation[1])] + tabu_list)[:tabu_length]
                elif operation[0] == '-':
                    current_model.remove_edge(*operation[1])
                    tabu_list = ([('+', operation[1])] + tabu_list)[:tabu_length]
                elif operation[0] == 'flip':
                    X, Y = operation[1]
                    current_model.remove_edge(X, Y)
                    current_model.add_edge(Y, X)
                    tabu_list = ([operation] + tabu_list)[:tabu_length]

            # hill climb
            print('----- hill climbing -----')
            current_model = self.estimate(start=current_model, tabu_list=tabu_list,
                                          tabu_length=tabu_length, max_indegree=max_indegree)
            current_score = K2Score(self.data).score(current_model)

            # compare with the best model
            if current_score > best_score:
                best_model = current_model
                best_score = current_score

            if len(self.lr_variables) > 0:
                self.lr_learnable.append(self.is_lr_learnable(current_model))

        return best_model.copy()

    def calculate_random_restart_length(self, i):
        return int(self.random_restart_length + i)

    def get_local_score(self, node, parents):
        local_score = self.scoring_method.local_score
        key = tuple([node, tuple(sorted(parents))])
        # get score from cache
        if key in self.scores[self.index].keys():
            return self.scores[self.index][key]
        # cache result for later use
        else:
            score = local_score(node, parents)
            self.scores[self.index][key] = score
            return score

    def is_lr_learnable(self, model):
        variable2lr = defaultdict(set)
        for i, lr in enumerate(self.lr_variables):
            for variable in lr:
                variable2lr[variable].add(i)

        # cross local-relation edges
        for start, end in model.edges:
            if variable2lr[start] & variable2lr[end] == set():
                return False

        # inbound edges from multiple tables
        inbound = defaultdict(set)
        for start, end in model.edges:
            inbound[end] |= (variable2lr[start] & variable2lr[end])
            if len(inbound[end]) > 1:
                return False

        return True


class GlobalHillClimbSearch(object):
    def __init__(self, parser, n_random_restarts=10, random_restart_length=5):
        """
        Class for heuristic hill climb searches for BayesianModels, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

        Parameters
        ----------
        parser: MultipleFileParser
        """

        self.parser = parser
        self.scoring_methods = []
        for single_parser in self.parser.single_file_parsers:
            self.scoring_methods.append(K2Score(single_parser.data_frame))

        # variable -> data source mapping
        self.variable_source_mapping = {}
        for i in range(len(self.parser.single_file_parsers)):
            parser = self.parser.single_file_parsers[i]
            for var in parser.variables:
                if var not in self.variable_source_mapping.keys():
                    self.variable_source_mapping[var] = {i}
                else:
                    self.variable_source_mapping[var].add(i)

        # random restart parameters
        self.n_random_restarts = n_random_restarts
        self.random_restart_length = random_restart_length

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        prohibited_edges = self.outbound_constraints(model)

        potential_new_edges = set()
        edge_map = {}
        for i in range(len(self.parser.single_file_parsers)):
            local_nodes = self.parser.single_file_parsers[i].variables
            potential_new_local_edges = (set(permutations(local_nodes, 2)) -
                                         set([(X, Y) for (X, Y) in model.edges()]) -
                                         set([(Y, X) for (X, Y) in model.edges()]) -
                                         prohibited_edges)

            # store which data source the edge resides in
            for edge in potential_new_local_edges:
                if edge in edge_map.keys():
                    edge_map[edge].append(i)
                else:
                    edge_map[edge] = [i]
            potential_new_edges.update(potential_new_local_edges)

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = list(model.get_parents(Y))
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_deltas = []
                        for index in edge_map[(X, Y)]:
                            nodes = set(old_parents + new_parents + [X, Y])
                            if len(list(filter(lambda x: x not in self.parser.single_file_parsers[index].variables,
                                               nodes))) > 0:
                                continue
                            local_score = self.scoring_methods[index].local_score
                            score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                            score_deltas.append(score_delta)
                        if len(score_deltas) > 0:
                            yield (operation, sum(score_deltas) / len(score_deltas))

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_deltas = []
                for index in self.data_source(X, Y):
                    nodes = set(old_parents + new_parents + [X, Y])
                    if len(list(
                            filter(lambda x: x not in self.parser.single_file_parsers[index].variables, nodes))) > 0:
                        continue
                    local_score = self.scoring_methods[index].local_score
                    score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                    score_deltas.append(score_delta)
                if len(score_deltas) > 0:
                    yield (operation, sum(score_deltas) / len(score_deltas))

        for (X, Y) in model.edges():  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)) and (Y, X) not in prohibited_edges:
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                    old_X_parents = list(model.get_parents(X))
                    old_Y_parents = list(model.get_parents(Y))
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        score_deltas = []
                        for index in self.data_source(X, Y):
                            nodes = set(old_X_parents + new_X_parents + old_Y_parents + new_Y_parents + [X, Y])
                            if len(list(filter(lambda x: x not in self.parser.single_file_parsers[index].variables,
                                               nodes))) > 0:
                                continue
                            local_score = self.scoring_methods[index].local_score
                            score_delta = (local_score(X, new_X_parents) +
                                           local_score(Y, new_Y_parents) -
                                           local_score(X, old_X_parents) -
                                           local_score(Y, old_Y_parents))
                            score_deltas.append(score_delta)
                        if len(score_deltas) > 0:
                            yield (operation, sum(score_deltas) / len(score_deltas))

    def _legal_operations_without_score(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        prohibited_edges = self.outbound_constraints(model)

        potential_new_edges = set()
        edge_map = {}
        for i in range(len(self.parser.single_file_parsers)):
            local_nodes = self.parser.single_file_parsers[i].variables
            potential_new_local_edges = (set(permutations(local_nodes, 2)) -
                                         set([(X, Y) for (X, Y) in model.edges()]) -
                                         set([(Y, X) for (X, Y) in model.edges()]) -
                                         prohibited_edges)

            # store which data source the edge resides in
            for edge in potential_new_local_edges:
                if edge in edge_map.keys():
                    edge_map[edge].append(i)
                else:
                    edge_map[edge] = [i]
            potential_new_edges.update(potential_new_local_edges)

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = list(model.get_parents(Y))
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        valid_count = 0
                        for index in edge_map[(X, Y)]:
                            nodes = set(old_parents + new_parents + [X, Y])
                            if len(list(filter(lambda x: x not in self.parser.single_file_parsers[index].variables,
                                               nodes))) > 0:
                                continue
                            valid_count += 1
                        if valid_count > 0:
                            yield operation

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                valid_count = 0
                for index in self.data_source(X, Y):
                    nodes = set(old_parents + new_parents + [X, Y])
                    if len(list(
                            filter(lambda x: x not in self.parser.single_file_parsers[index].variables, nodes))) > 0:
                        continue
                    valid_count += 1
                if valid_count > 0:
                    yield operation

        for (X, Y) in model.edges():  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list and (Y, X) not in prohibited_edges:
                    old_X_parents = list(model.get_parents(X))
                    old_Y_parents = list(model.get_parents(Y))
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        valid_count = 0
                        for index in self.data_source(X, Y):
                            nodes = set(old_X_parents + new_X_parents + old_Y_parents + new_Y_parents + [X, Y])
                            if len(list(filter(lambda x: x not in self.parser.single_file_parsers[index].variables,
                                               nodes))) > 0:
                                continue
                            valid_count += 1
                        if valid_count > 0:
                            yield operation

    def outbound_constraints(self, model):
        prohibited_edges = set()
        for (X, Y) in list(model.edges):
            if Y in self.parser.shared_variables:
                constrained_sources = self.variable_source_mapping[Y] - self.data_source(X, Y)
                for i in constrained_sources:
                    for var in self.parser.single_file_parsers[i].variables:
                        prohibited_edges.add((var, Y))
        return prohibited_edges

    def data_source(self, X, Y):
        """
        Finds the common data source between X and Y
        :param X:
        :param Y:
        :return: a list of indices
        """
        return self.variable_source_mapping[X].intersection(self.variable_source_mapping[Y])

    def estimate(self, start=None, tabu_list=[], tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: BayesianModel instance
            The starting point for the local search. By default a completely disconnected network is used.
        tabu_list: list
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        Returns
        -------
        model: `BayesianModel` instance
            A `BayesianModel` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data, scoring_method=BicScore(data))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """
        epsilon = 1e-8
        nodes = self.parser.relevant_variables
        if start is None:
            start = BayesianModel()
            start.add_nodes_from(nodes)
        elif not isinstance(start, BayesianModel) or not set(start.nodes()) == set(nodes):
            raise ValueError("'start' should be a BayesianModel with the same variables as the data set, or 'None'.")

        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model, tabu_list, max_indegree):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            print(best_operation)
            print(best_score_delta)

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]

        return current_model

    def global_score(self, model):
        score = 0
        for node in model.nodes():
            scores = []
            for index in self.variable_source_mapping[node]:
                nodes = list(filter(lambda x: x not in self.parser.single_file_parsers[index].variables,
                                    set([node] + list(model.predecessors(node)))))
                if len(nodes) > 0:
                    continue
                scores.append(self.scoring_methods[index].local_score(node, list(model.predecessors(node))))
            score += sum(scores) / len(scores)
        return score

    def random_restart(self, start=None, tabu_length=0, max_indegree=None):
        # starting best model
        if not start:
            best_model = self.estimate(tabu_length=tabu_length, max_indegree=max_indegree)
        else:
            best_model = start
        best_score = self.global_score(best_model)

        # iterate random restarts
        for i in range(self.n_random_restarts):
            current_model = best_model.copy()
            n_moves = i + self.random_restart_length
            tabu_list = []

            # perform random actions
            for j in range(n_moves):
                operations = []
                for operation in self._legal_operations_without_score(current_model, tabu_list, max_indegree):
                    operations.append(operation)

                try:
                    operation = random.choice(operations)
                except IndexError:
                    continue

                # perform operation
                if operation[0] == '+':
                    current_model.add_edge(*operation[1])
                    tabu_list = ([('-', operation[1])] + tabu_list)[:tabu_length]
                elif operation[0] == '-':
                    current_model.remove_edge(*operation[1])
                    tabu_list = ([('+', operation[1])] + tabu_list)[:tabu_length]
                elif operation[0] == 'flip':
                    X, Y = operation[1]
                    current_model.remove_edge(X, Y)
                    current_model.add_edge(Y, X)
                    tabu_list = ([operation] + tabu_list)[:tabu_length]

            # hill climbing
            print('----- hill climbing -----')
            current_model = self.estimate(start=current_model, tabu_list=tabu_list,
                                          tabu_length=tabu_length, max_indegree=max_indegree)
            current_score = self.global_score(current_model)

            # compare with the best model
            if current_score > best_score:
                best_model = current_model
                best_score = current_score

        return best_model.copy()
