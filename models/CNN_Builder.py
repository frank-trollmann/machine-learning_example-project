
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,  MaxPooling2D, Dropout, BatchNormalization, Flatten, Activation
from tensorflow.keras.constraints import UnitNorm

class CNN_Builder:
    """
        A builder class for a cnn.
    """

    def __init__(self,in_shape, out_shape, convolutional_layers, fully_connected_layers):
        """
            constructs a the builder object with a network structure. 
            Parameters:
                in_shape -- the shape of the input of the network
                out_shape -- the shape of the output of the network
                convolutional_layers -- the number of patterns for the convolutional layers. [2,3] will create two convolutional layers with 2 and 3 patterns
                fully_connected_layers -- the number of neurons for the fully connected layers. [100,200] will create two fully connected layers with 100 and 200 neurons
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.convolutional_layers = convolutional_layers
        self.fully_connected_layers = fully_connected_layers

        # initialize additional configuration as false. Can be set from the outside even between calls of build_model
        self.apply_regularization = False
        self.apply_dropout = False
        self.apply_batch_normalization = False
        self.weight_constraints = False

    def build_model(self):
        """
            Build a model based on the configuration of the builder.
            Each call builds a fresh model with different weights. 
        """
        model = Sequential()

        for index in range(len(self.convolutional_layers)):
            self.add_convolutional_layer(model=model, filters=self.convolutional_layers[index], first_layer=index == 0)

        model.add(Flatten())       
                                
        for index in range(len(self.fully_connected_layers)):
            self.add_fully_connected_layer(model=model, nr_neurons=self.fully_connected_layers[index])

        model.add(Dense(self.out_shape,activation="softmax"))
        model.compile(loss="binary_crossentropy", optimizer='adam',run_eagerly=True)

        return model

    def add_convolutional_layer(self,model, filters, first_layer):
        """
            adds one convolutional layer stack to the model.
            This includes a convolutional layer, activation and pooling layer.
            May also include additional regularization / normalization operations according to the class configuration.

            Dropout is not applied, following the advise in https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
        """
        model_configuration = {}

        if first_layer:
            model_configuration["input_shape"] = self.in_shape

        if self.weight_constraints :
            model_configuration["kernel_constraint"] = UnitNorm() 

        if self.apply_regularization:
            model_configuration["kernel_regularizer"] = "l2"
        
        model.add(Conv2D(filters,
                            (3,3),
                            activation = "tanh",
                            padding = "same", 
                            **model_configuration))          

        if self.apply_batch_normalization:
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2)))

        
    
    def add_fully_connected_layer(self,model,nr_neurons):
        """
            adds one fully connected layer. This includes a dropout layer and other normalization operations according to the class configuration.
        """
        model.add(Dense(nr_neurons,activation ="tanh"))
        if self.apply_batch_normalization:
            model.add(BatchNormalization())

        if self.apply_dropout:
            model.add(Dropout(0.5))
