## ResNet50 for binary image classification
This branch contains a simple console program in Python to train, evaluate and save ResNet50 model on specified image dataset.

Script needs Tensorflow, scikit-learn, PIL and matplotlib libraries installed in order to be successfully run.
```
-data <path> = specify path to dataset
-o <path> = output path (must be a directory)
-epochs <count> = set epoch count (default is 5)
-target <class_name> = set target class (based on classes in input data)
-isz <WxH> = set image size for model
-nogpu = don't use GPU for training/testing, does nothing on systems with no GPU
```
