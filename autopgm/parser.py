import pandas


class SingleFileParser(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_frame = pandas.read_csv(self.file_name)
        self.variables = self.data_frame.columns.values.tolist()
        self.shared_variables = set()

    def filter(self, variables):
        variables = list(filter(lambda x: x in self.variables, variables))
        self.data_frame = self.data_frame[variables]
        self.variables = self.data_frame.columns.values.tolist()
        return self.data_frame

    def populate_shared_variables(self, variables):
        self.shared_variables = set(filter(lambda x: x in self.variables, variables))


class MultipleFileParser(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.shared_variable_dict = {}
        self.shared_dict = {}
        self.orientation_candidates = {}
        self.orientations = []

        # parse all files
        self.single_file_parsers = list(map(lambda x: SingleFileParser(x), self.file_names))
        self.variables = set()
        for file_parser in self.single_file_parsers:
            self.variables.update(file_parser.variables)

        # update variables
        self.variables = set()
        for file_parser in self.single_file_parsers:
            self.variables.update(file_parser.variables)

        # shared variables
        all_variables = []
        temp_variables = set()
        self.shared_variables = set()
        for file_parser in self.single_file_parsers:
            all_variables.extend(file_parser.variables)
        for variable in all_variables:
            if variable not in temp_variables:
                temp_variables.add(variable)
            else:
                self.shared_variables.add(variable)

        # populate shared variables
        for file_parser in self.single_file_parsers:
            file_parser.populate_shared_variables(self.shared_variables)

        # orientations
        self.calculate_orientations()

    def calculate_orientations(self):
        # orientations of shared variables
        for i in range(len(self.single_file_parsers)):
            variables = self.single_file_parsers[i].variables
            for variable in variables:
                if variable not in self.shared_variable_dict.keys():
                    self.shared_variable_dict[variable] = [i]
                else:
                    self.shared_variable_dict[variable].append(i)

        for variable in self.shared_variable_dict.keys():
            if len(self.shared_variable_dict[variable]) > 1:
                self.shared_dict[variable] = self.shared_variable_dict[variable][:]

        # permutations of orientations
        for variable in self.shared_dict.keys():
            self.orientation_candidates[variable] = []
            # all outbound
            current_orientation = {}
            for position in self.shared_dict[variable]:
                current_orientation[position] = 0
            self.orientation_candidates[variable].append(current_orientation)
            # one inbound, rest outbound
            for i in self.shared_dict[variable]:
                current_orientation = {}
                for position in self.shared_dict[variable]:
                    if position == i:
                        current_orientation[position] = 1
                    else:
                        current_orientation[position] = 0
                self.orientation_candidates[variable].append(current_orientation)

        # orientation options
        for variable in self.orientation_candidates.keys():
            if len(self.orientations) == 0:
                for orientation in self.orientation_candidates[variable]:
                    self.orientations.append({variable: orientation})
            else:
                new_orientations = []
                for old_orientation in self.orientations:
                    for new_orientation in self.orientation_candidates[variable]:
                        orientation_copy = old_orientation.copy()
                        orientation_copy[variable] = new_orientation
                        new_orientations.append(orientation_copy)
                self.orientations = new_orientations
