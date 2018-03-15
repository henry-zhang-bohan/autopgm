import numpy as np
import pandas
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from autopgm.external.HillClimbSearch import *


class Node(object):
    def __init__(self, name, parents):
        self.name = name
        self.visited = False
        self.parents = parents


class PGMGraph(object):
    def __init__(self, model):
        self.graph = {}

        # populate adjacency lists
        for cpd in model.get_cpds():
            child = cpd.variable
            parents = cpd.variables[1:]

            if child not in self.graph.keys():
                self.graph[child] = Node(child, parents)
            else:
                self.graph[child].parents.extend(parents)

        # number of nodes
        self.N = len(self.graph)

    # A recursive function used by topological_sort
    def topological_sort_util(self, node, visited, stack):

        # Mark the current node as visited.
        visited[node] = True

        # Recur for all the vertices adjacent to this vertex
        for parent in self.graph[node].parents:
            if not visited[parent]:
                self.topological_sort_util(parent, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, node)

    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = {}
        for node in self.graph.keys():
            visited[node] = False
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for node in self.graph.keys():
            if not visited[node]:
                self.topological_sort_util(node, visited, stack)

        return list(reversed(stack))


class CSVWriter(object):
    def __init__(self, model, file_name, size=10000):
        self.model = model
        self.size = size
        self.file_name = file_name
        self.graph = PGMGraph(self.model)
        self.order = self.graph.topological_sort()

        # populate cpd's
        self.cpds = {}
        for cpd in self.model.get_cpds():
            self.cpds[cpd.variable] = cpd

        # initialize data
        self.data = np.zeros((self.size, len(self.order)), dtype=np.int)
        self.populate_data()

        # write to csv
        self.write_to_csv()

    def populate_data(self):
        for j in range(len(self.order)):
            col = self.order[j]
            for i in range(self.size):
                self.data[i, j] = self.calculate_single_data(col, i)

    def calculate_single_data(self, col, i):
        parents = self.cpds[col].variables[1:]
        parents_value = []
        for parent in parents:
            parents_value.append((parent, int(self.data[i, self.order.index(parent)])))

        # get probability distribution
        cpd_copy = self.cpds[col].copy()
        cpd_copy.reduce(parents_value)
        reduced_prob = cpd_copy.get_values().flatten()
        accumulated_prob = []
        for i in range(len(reduced_prob)):
            accumulated_prob.append(sum(reduced_prob[:i + 1]))
        accumulated_prob.insert(0, 0)

        # generate prediction of state
        state = -1
        random_number = random.random()
        for i in range(1, len(accumulated_prob)):
            if accumulated_prob[i - 1] <= random_number < accumulated_prob[i]:
                state = i - 1
                break

        return state

    def write_to_csv(self):
        np.savetxt(self.file_name, self.data, fmt='%i', delimiter=',', header=','.join(self.order), comments='')


class CSVSplitter(object):
    def __init__(self, file_name, table_columns, pre, split_path=''):
        self.data_frame = pandas.read_csv(file_name)
        for i in range(len(table_columns)):
            data_frame = self.data_frame[list(set(table_columns[i]))]
            data_frame.to_csv(split_path + pre + '_' + str(i + 1) + '.csv', index=False)


class TrainTestSplitter(object):
    def __init__(self, file_name, pre, frac=0.8, split_path=''):
        data_frame = pandas.read_csv(file_name).sample(frac=1., random_state=0)
        split_index = int(frac * data_frame.shape[0])
        train_df = data_frame[:split_index]
        test_df = data_frame[split_index:]
        train_df.to_csv(split_path + pre + '_train.csv', index=False)
        test_df.to_csv(split_path + pre + '_test.csv', index=False)


class ModelVerifier(object):
    def __init__(self, file_name, edges=None):
        self.file_name = file_name
        self.data_frame = pandas.read_csv(file_name)
        self.edges = edges
        self.model = BayesianModel(edges)
        self.model.fit(self.data_frame, estimator=MaximumLikelihoodEstimator)

    def get_cpds(self):
        return self.model.get_cpds()

    def print_cpds(self):
        for cpd in self.model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)


class StudentModel1(object):
    def __init__(self):
        student_model_1 = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

        grade_cpd = TabularCPD(
            variable='G',
            variable_card=3,
            values=[[0.3, 0.05, 0.9, 0.5],
                    [0.4, 0.25, 0.08, 0.3],
                    [0.3, 0.7, 0.02, 0.2]],
            evidence=['I', 'D'],
            evidence_card=[2, 2])

        difficulty_cpd = TabularCPD(
            variable='D',
            variable_card=2,
            values=[[0.6, 0.4]])

        intel_cpd = TabularCPD(
            variable='I',
            variable_card=2,
            values=[[0.7, 0.3]])

        letter_cpd = TabularCPD(
            variable='L',
            variable_card=2,
            values=[[0.1, 0.4, 0.99],
                    [0.9, 0.6, 0.01]],
            evidence=['G'],
            evidence_card=[3])

        sat_cpd = TabularCPD(
            variable='S',
            variable_card=2,
            values=[[0.95, 0.2],
                    [0.05, 0.8]],
            evidence=['I'],
            evidence_card=[2])

        student_model_1.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, letter_cpd, sat_cpd)
        self.model = student_model_1


class StudentModel2(object):
    def __init__(self):
        student_model_2 = BayesianModel([('D', 'P'), ('I', 'P')])

        difficulty_cpd = TabularCPD(
            variable='D',
            variable_card=2,
            values=[[0.6, 0.4]])

        intel_cpd = TabularCPD(
            variable='I',
            variable_card=2,
            values=[[0.7, 0.3]])

        pass_cpd = TabularCPD(
            variable='P',
            variable_card=2,
            values=[[0.3, 0.8, 0.1, 0.4],
                    [0.7, 0.2, 0.9, 0.6]],
            evidence=['I', 'D'],
            evidence_card=[2, 2])

        student_model_2.add_cpds(difficulty_cpd, intel_cpd, pass_cpd)
        self.model = student_model_2


class StudentModel3(object):
    def __init__(self):
        student_model_3 = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S'),
                                         ('D', 'P'), ('I', 'P'), ('P', 'L')])

        grade_cpd = TabularCPD(
            variable='G',
            variable_card=3,
            values=[[0.3, 0.05, 0.9, 0.5],
                    [0.4, 0.25, 0.08, 0.3],
                    [0.3, 0.7, 0.02, 0.2]],
            evidence=['I', 'D'],
            evidence_card=[2, 2])

        difficulty_cpd = TabularCPD(
            variable='D',
            variable_card=2,
            values=[[0.6, 0.4]])

        intel_cpd = TabularCPD(
            variable='I',
            variable_card=2,
            values=[[0.7, 0.3]])

        letter_cpd = TabularCPD(
            variable='L',
            variable_card=2,
            values=[[0.2, 0.5, 0.999, 0.1, 0.4, 0.99],
                    [0.8, 0.5, 0.001, 0.9, 0.6, 0.01]],
            evidence=['P', 'G'],
            evidence_card=[2, 3])

        sat_cpd = TabularCPD(
            variable='S',
            variable_card=2,
            values=[[0.95, 0.2],
                    [0.05, 0.8]],
            evidence=['I'],
            evidence_card=[2])

        pass_cpd = TabularCPD(
            variable='P',
            variable_card=2,
            values=[[0.3, 0.8, 0.1, 0.4],
                    [0.7, 0.2, 0.9, 0.6]],
            evidence=['I', 'D'],
            evidence_card=[2, 2])

        student_model_3.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, letter_cpd, sat_cpd, pass_cpd)
        self.model = student_model_3
