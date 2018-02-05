import numpy as np
import random
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


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
    def __init__(self, model, file_name, size=1000):
        self.model = model
        self.size = size
        self.file_name = file_name
        self.graph = PGMGraph(self.model)
        self.order = self.graph.topological_sort()

        # populate cpds
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

        # generate prediction of state
        random_number = random.random()
        state = 0
        for i in range(len(reduced_prob)):
            if random_number < reduced_prob[i]:
                state = i

        return state

    def write_to_csv(self):
        np.savetxt(self.file_name, self.data, fmt='%i', delimiter=',', header=','.join(self.order), comments='')


class StudentModel(object):
    def __init__(self):
        student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

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

        student_model.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, letter_cpd, sat_cpd)
        self.model = student_model