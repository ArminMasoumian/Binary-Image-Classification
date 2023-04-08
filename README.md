# Binary Image Classification using Convolutional Neural Networks (CNNs)

This repository contains an implementation of a binary image classification model using convolutional neural networks (CNNs) in PyTorch. The model is trained and evaluated on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The task is to classify each image as either a cat or a dog.

## Requirements

Python 3.7 or higher

PyTorch 1.9 or higher

torchvision 0.10 or higher

## Usage

1. Clone the repository:

```
git clone https://github.com/username/binary-image-classification.git
```

2. Navigate to the cloned repository:

```
cd binary-image-classification
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Download the CIFAR-10 dataset:

```
python download_dataset.py
```

5. Train the model:

```
python train.py
```

6. Evaluate the model:

```
python evaluate.py
```

##Results

The model achieved an accuracy of 98.3% on the test set after training for 50 epochs. The training and validation curves are shown below:

## References

CIFAR-10 dataset
PyTorch documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
