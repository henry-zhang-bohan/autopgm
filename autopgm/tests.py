from test_data_generator import *
from merger import *

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
