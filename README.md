# covid-cnn

A convolutional neural netowrk that detects covid-19 and pneumonia from an x-ray scan of the lungs. This repo contains a pretrained model already (see below for more specifics on how it performs)

## Accuracy of pretrained model

Using the normalized training dataset already compiled in this repo (`training-data/editied-images`), this model has around a 92% accuracy.

```
**Traing Example (batch size is 32, 2 epochs)**
Epoch 1/2
161/161 [==============================] - 78s 486ms/step - loss: 0.4481 - sparse_categorical_accuracy: 0.8248

Epoch 2/2
161/161 [==============================] - 73s 451ms/step - loss: 0.2150 - sparse_categorical_accuracy: 0.9236

**Testing Example (batch size is 32)** 
41/41 [==============================] - 4s 97ms/step - loss: 0.2377 - sparse_categorical_accuracy: 0.9154
```

Normalized testing dataset contains:
117 covid images
318 normal images
856 pneumonia images


Normalized training dataset contains:
461 covid images
1,267 normal images
3,419 pnemonia images

## Commands

To create training data in an already populated `training-data/` folder, run
`python3 main.py create_training_data`

To create testing data in an already populated `testing-data/` folder, run
`python3 main.py create_testing_data`

If you already created training data, you can train a new model with
`python3 main.py train_new_model`

If you already have a trained model and would like to train it more, you can train your model with
`python3 main.py train_old_model`

If you already have a trained model and have created your testing data, you can test your model with
`python3 main.py test_model`

## Use your own dataset and model
Before you can use your own datasets (for testing and training) you have to clear a lot of files. I made a convenience script clear all the necessary directories, so just run `./clear_all.sh` (make sure you are in the root folder of this repo). If you get a permission denied as a result of running the script, you may need to run this command first: `chmod +x clear_all.sh`.



