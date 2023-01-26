import unittest
from models.cnn_builder import CNNBuilder

from keras.layers.convolutional.conv2d import Conv2D
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras.layers.reshaping.flatten import Flatten
from keras.layers.core.dense import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout


class TestCNNBuilder(unittest.TestCase):
    """
        tests the CNN builder
    """

    def setUp(self):
        self.cnn_builder = CNNBuilder(
            convolutional_layers=[77,777],
            fully_connected_layers=[8,9],
            in_shape=(100,200,3),    
            out_shape=7
        )
        self.cnn_builder.apply_regularization = True
        self.cnn_builder.apply_dropout = True
        self.cnn_builder.apply_batch_normalization = True
        self.cnn_builder.weight_constraints = True
        self.model = self.cnn_builder.build_model()

    def test_input_shape(self):
        """
            tests that the created model has the correct input shape
        """
        assert self.model.input.shape[1] == 100
        assert self.model.input.shape[2] == 200
        assert self.model.input.shape[3] == 3
    
    def test_output_shape(self):
        """
            tests that the created model has the correct output shape
        """
        assert self.model.output.shape[1] == 7
    
    def test_cnn_layers(self):
        """
            tests that the CNN part of the network is generated correctly
        """

        assert type(self.model.layers[0]) == Conv2D
        assert self.model.layers[0].filters == 77

        assert type(self.model.layers[1]) == BatchNormalization
        
        assert type(self.model.layers[2]) == MaxPooling2D
        
        assert type(self.model.layers[3]) == Conv2D
        assert self.model.layers[3].filters == 777

        assert type(self.model.layers[4]) == BatchNormalization

        assert type(self.model.layers[5]) == MaxPooling2D
    
    def test_flatten_layer(self):
        """
            tests that the cnn and ann part of the network are separted with a flattening layer
        """
        assert type(self.model.layers[6]) == Flatten
    
    def test_ann_layers(self):
        """
            tests that the ANN part of the network is generated correctly
        """
        assert type(self.model.layers[7]) == Dense
        assert self.model.layers[7].units == 8

        assert type(self.model.layers[8]) == BatchNormalization

        assert type(self.model.layers[9]) == Dropout

        assert type(self.model.layers[10]) == Dense
        assert self.model.layers[10].units == 9

        assert type(self.model.layers[11]) == BatchNormalization

        assert type(self.model.layers[12]) == Dropout


        assert type(self.model.layers[13]) == Dense
        assert self.model.layers[13].units == 7


    def tearDown(self):
        pass
