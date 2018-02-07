from generator import *
from parser import *
from estimator import *
from pgmpy.inference import VariableElimination

DATA_DIR = 'data/'

# test_data_generator
TEST_DATA_GENERATOR_MODEL = True
if TEST_DATA_GENERATOR_MODEL:
    sm1 = StudentModel1().model
    sm2 = StudentModel2().model

TEST_DATA_GENERATOR_WRITE = False
if TEST_DATA_GENERATOR_WRITE:
    csv_w1 = CSVWriter(sm1, DATA_DIR + 'student1.csv')
    print('Created Student Model 1 CSV.')
    csv_w1 = CSVWriter(sm2, DATA_DIR + 'student2.csv')
    print('Created Student Model 2 CSV.')

TEST_DATA_GENERATOR_VERIFY = False
if TEST_DATA_GENERATOR_VERIFY:
    print('---------- STUDENT MODEL 1 CPDS ----------')
    mv1 = ModelVerifier(DATA_DIR + 'student1.csv', sm1.edges)
    mv1.print_cpds()

    print('---------- STUDENT MODEL 2 CPDS ----------')
    mv1 = ModelVerifier(DATA_DIR + 'student2.csv', sm2.edges)
    mv1.print_cpds()

# merger
MERGER = False
if MERGER:
    sm = BayesianMerger([sm1, sm2]).merge()
    if sm:
        for cpd in sm.get_cpds():
            print(cpd)
    else:
        print('Model not valid.')

# parser
PARSER = True
if PARSER:
    file_names = [
        DATA_DIR + 'student1.csv',
        DATA_DIR + 'student2.csv'
    ]
    known_edges = [('I', 'G')]
    query_targets = ['L', 'P']
    query_evidence = ['I', 'D']
    mfp = MultipleFileParser(file_names, known_edges, query_targets, query_evidence)

# hill climb search
HILL_CLIMB_SEARCH = True
if HILL_CLIMB_SEARCH:
    mbe = MultipleBayesianEstimator(mfp)
    sm = mbe.merged_model

    # edges
    print('Edges:', sm.edges)
    print('--------------------')

    # print cpds
    for cpd in sm.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)

    inference = VariableElimination(sm)

    # SAT samples
    print('----- SAT: Unknown -----')
    print(inference.query(['P'])['P'])
    print('----- SAT: 0 -----')
    print(inference.query(['P'], evidence={'S': 0})['P'])
    print('----- SAT: 1 -----')
    print(inference.query(['P'], evidence={'S': 1})['P'])

    # Letter samples
    print('----- Letter: Unknown -----')
    print(inference.query(['P'])['P'])
    print('----- Letter: 0 -----')
    print(inference.query(['P'], evidence={'L': 0})['P'])
    print('----- Letter: 1 -----')
    print(inference.query(['P'], evidence={'L': 1})['P'])
