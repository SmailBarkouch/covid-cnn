# covid-cnn

A convolutional neural netowrk that detects covid-19 and pneumonia from an x-ray scan of the lungs. This model is already trained.

## Commands

To create training data in an already populated `training-data/` folder:
`python3 main.py create_training_data`

To create testing data in an already populated `testing-data/` folder:
`python3 main.py create_testing_data`

If you already created training data, you can train a new model with:
`python3 main.py train_new_model`

If you already have a trained model and would like to train it more, you can train it with:
`python3 main.py train_old_model`

If you already have a trained model and have created your testing data, you can test it with:
`python3 main.py test_model`

## Use your own dataset and model

If you would like to train your own model with your own data set, first clear out all the training images (but not the folders) in `training-data/`, `testing-data`. Then clear the already trained model, in the `trained-model` folder (get rid of everything in it, including the folders).

