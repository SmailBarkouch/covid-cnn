import sys
from train import create_training_data, load_training_data
from model import create_model
from sample import create_testing_data, load_testing_data

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
elif sys.argv[1] == "create_model":
    create_model(load_training_data())
else:
    print("Your argument doesn't match any available method.")

