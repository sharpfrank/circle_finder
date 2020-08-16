# Circle Finder 

Find the location and radius of a circle in a noisy image.

## Model Training and Testing

### Training

```python train_model.py --epochs 10 --batch_size 256 --train_size 10000 --val_size 2000 --model_name circle_model.pk -- model_type cireclenet18```

### Testing

```python test_model.py --model_name circle_model.pk --model_type circlenet18 --test_size 1000```

## Directories

### circle

* ```circle_dataset.py``` - A pytorch Dataset for sourcing training images.
* ```circle_find.py``` - Defines the circle_find function used to test the circle detection model.
* ```circle_generator.py``` - Functions for creating circle images and measuring there overlap.
* ```circle_train.py``` - A function for training the circle detection model.
* ```circlenet.py``` - Defines the model class used to detect circles.

### models

Trained models are stored here. 

```circlenet18-d200k40k.pk``` is the required model file. It was trained with ```train_model.py``` and is tested with ```test_model.py```.

### results

The log files for training and the testing log files are stored here.

```circlenet18-d200k40k_output.txt``` is the required output training log file.
