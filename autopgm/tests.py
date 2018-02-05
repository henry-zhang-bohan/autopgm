import test_data_generator

DATA_DIR = 'data/'

TEST_DATA_GENERATOR_MODEL = True
if TEST_DATA_GENERATOR_MODEL:
    sm1 = test_data_generator.StudentModel1().model
    sm2 = test_data_generator.StudentModel2().model

TEST_DATA_GENERATOR_WRITE = False
if TEST_DATA_GENERATOR_WRITE:
    csv_w1 = test_data_generator.CSVWriter(sm1, DATA_DIR + 'student1.csv')
    print('Created Student Model 1 CSV.')
    csv_w1 = test_data_generator.CSVWriter(sm2, DATA_DIR + 'student2.csv')
    print('Created Student Model 2 CSV.')

TEST_DATA_GENERATOR_VERIFY = True
if TEST_DATA_GENERATOR_VERIFY:
    print('---------- STUDENT MODEL 1 CPDS ----------')
    mv1 = test_data_generator.ModelVerifier(DATA_DIR + 'student1.csv', sm1.edges)
    mv1.print_cpds()

    print('---------- STUDENT MODEL 2 CPDS ----------')
    mv1 = test_data_generator.ModelVerifier(DATA_DIR + 'student2.csv', sm2.edges)
    mv1.print_cpds()
