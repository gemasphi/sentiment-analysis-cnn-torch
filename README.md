# Sentiment Analysis with CNNs in pytorch

A sentiment analysis implementation with CNNs done in pytorch.

The code structure is as follows:
  1. **main.py:** contains the model config, trains and tests the model ploting the loss funcion and saving the results to a cvs file.
  2. **model.py:** contains the model based on [1]
  3. **readers.py:** contains a class to read the standford sentiment treebank dataset using [2] and class to read glove embeddings
  5. **train.py:** trains a model
  4. **test.py:** tests a trained model


#### References:
1. [Convolutional Neural Networks for Sentence Classification, Kim, Yoon, 2014](https://arxiv.org/pdf/1408.5882.pdf) 
2. [pytreebank](https://github.com/JonathanRaiman/pytreebank)
