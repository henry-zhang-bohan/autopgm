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
JOBS_EXP = False
if JOBS_EXP:
    import pickle

    JOBS_EXP_TRAIN = True
    if JOBS_EXP_TRAIN:
        print('---------- JOBS EXP CPDS ----------')
        mbe = MultipleBayesianEstimator([DATA_DIR + 'jobs.csv'],
                                        query_targets=['happy', 'financial'],
                                        query_evidence=['em', 'income', 'ownrent'],
                                        outbound_nodes=['sex'],
                                        known_independencies=[],
                                        n_random_restarts=10, random_restart_length=0)
        pickle.dump(mbe, open(DATA_DIR + 'jobs_exp.p', 'wb'))

    mbe = pickle.load(open(DATA_DIR + 'jobs_exp.p', 'rb'))
    mbe.print_edges()
    mbe.print_cpds()
    mbe.plot_edges()

# edu.csv experiment
EDU_EXP = False
if EDU_EXP:
    import pickle

    EDU_EXP_TRAIN = True
    if EDU_EXP_TRAIN:
        print('---------- EDU EXP CPDS ----------')
        mbe = MultipleBayesianEstimator([DATA_DIR + 'edu.csv'], query_targets=['income', 'happy'],
                                        query_evidence=['age', 'irace', 'marital', 'sex', 'relig', 'attend',
                                                        'hh1', 'family', 'financial', 'party', 'ideo', 'state',
                                                        'ownrent', 'job_sat', 'equality', 'edu', 'eu_edu_use', 'em',
                                                        'look_4_em'],
                                        outbound_nodes=['age', 'irace', 'sex'],
                                        known_independencies=[],
                                        n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe, open(DATA_DIR + 'edu_exp.p', 'wb'))

    mbe = pickle.load(open(DATA_DIR + 'edu_exp.p', 'rb'))
    mbe.print_edges()
    mbe.print_cpds()
    mbe.plot_edges()

# edu.csv experiment
EDU_EXP2 = True
if EDU_EXP2:
    import pickle

    EDU_EXP_TRAIN = False
    if EDU_EXP_TRAIN:
        print('---------- EDU EXP CPDS ----------')
        mbe = MultipleBayesianEstimator([DATA_DIR + 'edu.csv'], query_targets=['income', 'happy'],
                                        query_evidence=['marital',  'ownrent', 'job_sat', 'em', 'look_4_em'],
                                        outbound_nodes=['sex'],
                                        known_independencies=[],
                                        n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe, open(DATA_DIR + 'edu_exp2.p', 'wb'))

    EDU_EXP_SHOW = False
    if EDU_EXP_SHOW:
        mbe = pickle.load(open(DATA_DIR + 'edu_exp2.p', 'rb'))
        mbe.print_edges()
        mbe.print_cpds()
        mbe.plot_edges()

    EDU_SUB = False
    if EDU_SUB:
        print('---------- EDU EXP SUB 1 ----------')
        mbe1 = MultipleBayesianEstimator([DATA_DIR + 'edu.csv'], query_targets=['income'],
                                         query_evidence=['marital', 'sex', 'ownrent', 'em', 'look_4_em'],
                                         outbound_nodes=['sex'],
                                         known_independencies=[],
                                         n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe1, open(DATA_DIR + 'edu_exp2_1.p', 'wb'))

        print('---------- EDU EXP SUB 2 ----------')
        mbe2 = MultipleBayesianEstimator([DATA_DIR + 'edu.csv'], query_targets=['income'],
                                         query_evidence=['edu', 'eu_edu_use'],
                                         outbound_nodes=['sex'],
                                         known_independencies=[],
                                         n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe2, open(DATA_DIR + 'edu_exp2_2.p', 'wb'))

        print('---------- EDU EXP SUB 3 ----------')
        mbe3 = MultipleBayesianEstimator([DATA_DIR + 'edu.csv'], query_targets=['happy'],
                                         query_evidence=['em', 'look_4_em', 'job_sat', 'financial', 'family'],
                                         outbound_nodes=['sex'],
                                         known_independencies=[],
                                         n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe3, open(DATA_DIR + 'edu_exp2_3.p', 'wb'))

    EDU_SHOW_SUB = False
    if EDU_SHOW_SUB:
        mbe1 = pickle.load(open(DATA_DIR + 'edu_exp2_1.p', 'rb'))
        mbe2 = pickle.load(open(DATA_DIR + 'edu_exp2_2.p', 'rb'))
        mbe3 = pickle.load(open(DATA_DIR + 'edu_exp2_3.p', 'rb'))

    EDU_EXP_MERGE = True
    if EDU_EXP_MERGE:
        print('---------- EDU EXP MERGE CPDS ----------')
        file_names = [DATA_DIR + 'edu_1.csv', DATA_DIR + 'edu_2.csv']
        mbe = MultipleBayesianEstimator(file_names, query_targets=['income', 'happy'],
                                        query_evidence=['irace', 'marital', 'sex', 'relig', 'family', 'financial',
                                                        'ownrent', 'job_sat', 'edu', 'eu_edu_use', 'em', 'look_4_em'],
                                        outbound_nodes=['irace', 'sex'],
                                        known_independencies=[],
                                        n_random_restarts=0, random_restart_length=0)
        pickle.dump(mbe, open(DATA_DIR + 'edu_exp2_merged.p', 'wb'))

    EDU_EXP_SHOW = False
    if EDU_EXP_SHOW:
        mbe = pickle.load(open(DATA_DIR + 'edu_exp2_merged.p', 'rb'))
        mbe.print_edges()
        mbe.print_cpds()
        mbe.plot_edges()
