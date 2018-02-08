from generator import *
from estimator import *
from pgmpy.inference import VariableElimination

DATA_DIR = 'data/'

# test_data_generator
TEST_DATA_GENERATOR_MODEL = False
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
    mv2 = ModelVerifier(DATA_DIR + 'student2.csv', sm2.edges)
    mv2.print_cpds()

# merger
MERGER = False
if MERGER:
    sm = BayesianMerger([sm1, sm2], [10000, 20000]).merge()
    if sm:
        for cpd in sm.get_cpds():
            print(cpd)
    else:
        print('Model not valid.')

# hill climb search
HILL_CLIMB_SEARCH = False
if HILL_CLIMB_SEARCH:
    file_names = [
        DATA_DIR + 'student1.csv',
        DATA_DIR + 'student2.csv'
    ]
    mbe = MultipleBayesianEstimator(file_names, n_random_restarts=0, random_restart_length=0, start=None)
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

# jobs.csv experiment
JOBS_EXP = True
if JOBS_EXP:
    import pickle
    JOBS_EXP_TRAIN = True
    if JOBS_EXP_TRAIN:
        print('---------- JOBS EXP CPDS ----------')
        mbe = MultipleBayesianEstimator([DATA_DIR + 'jobs.csv'], query_targets=['income', 'happy'],
                                        query_evidence=['age', 'racethn', 'marital', 'parent', 'sex', 'relig', 'attend',
                                                        'family', 'financial', 'party', 'ideo', 'state', 'ownrent'],
                                        outbound_nodes=['age', 'racethn', 'sex'],
                                        known_independencies=[('marital', 'attend')],
                                        n_random_restarts=5, random_restart_length=0)
        pickle.dump(mbe, open(DATA_DIR + 'jobs_exp.p', 'wb'))

    mbe = pickle.load(open(DATA_DIR + 'jobs_exp.p', 'rb'))
    mbe.print_edges()
    mbe.print_cpds()
    mbe.plot_edges()

    # [('age', 'ownrent'), ('age', 'parent'), ('ownrent', 'marital'), ('ownrent', 'income'), ('marital', 'income'),
    # ('income', 'financial'), ('family', 'happy'), ('sex', 'marital'), ('relig', 'attend'), ('relig', 'ideo'),
    # ('relig', 'party'), ('racethn', 'relig'), ('racethn', 'state'), ('racethn', 'party'), ('financial', 'family'),
    # ('financial', 'happy'), ('party', 'ideo'), ('parent', 'marital')]


