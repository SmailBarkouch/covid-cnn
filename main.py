import sys
from train import create_training_data, load_training_data, train_model, load_trained_model
from model import create_model
from test import create_testing_data, load_testing_data, test_model

# Why the fudge doesn't python have pattern matching

if len(sys.argv) == 0:
    print("Please provide an argument that relates to one of the methods.")
    exit
elif len(sys.argv) > 2:
    print("Please provide one argument.")
    exit

if sys.argv[1] == "create_training_data":
    create_training_data()
elif sys.argv[1] == "create_testing_data":
    create_testing_data()
elif sys.argv[1] == "train_new_model":
    training_data = load_training_data()
    train_model(create_model(training_data), training_data, 2)
elif sys.argv[1] == "train_old_model":
    training_data = load_training_data()
    train_model(load_trained_model(), training_data, 2)
elif sys.argv[1] == "test_model":
    testing_data = load_testing_data()
    test_model(load_trained_model(), testing_data)
else:
    print("Your argument doesn't match any available method.")

