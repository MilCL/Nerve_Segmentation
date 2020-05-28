import numpy as np
import tensorflow as tf
import warnings
import os
import pickle
import matplotlib.image as mpimg
import uuid
from PIL import Image
from training_plots import *
from util_images import get_annotated_data
from util_images import *
from metrics import *

pix = 96
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../models/')

def predict_example_and_plot(model, X, Y, size):
    for i in range(len(X)):
        Y_pred = model.predict(X[i].reshape((1,pix, pix,1))) > 0.5
        plot_image_with_mask(X[i], Y[i], pred_mask=Y_pred, size = pix)
    return



class UNET():

    def __init__(self,
                 architecture = [1024,512,256,128,64],
                 img_dims = (544,544,1),
                 lambd = 1e-5,
                 dropout = False,
                 loss_f = dice_coef_loss,
                 metrics = [],
                 param_conv = { 'dropout': False,
                                'activation' : 'relu',
                                'kernel_initializer' : 'he_normal',
                                'padding' : 'same'}, weight_factor=0
                 ):
        
        
        self.img_dims = img_dims  
#        Input dimensions
        
        self.param_conv = param_conv
#        Parameters and charateristics of the convolution layer
        
        self.architecture = {'encoding_path': list(reversed(architecture))[:-1],
                             'bottom': architecture[0],
                             'decoding_path': architecture[1:]}
#        Architecture of the network
        
        self.optimizer = tf.keras.optimizers.Adam(lambd)
#        Optimizer 
        
        self.lambd = lambd
#        Initial learning rate used by the optimizer
        
        self.depth = len(architecture)
#        Depth of the network
        
        self.loss_function = loss_f
#        Loss function, the loss funtion are implemented in metrics.py
        
        self.is_trained = False
#       is True if network has been trained, false otherwise
        
        self.metrics = metrics
#        List of metrics, implemented in metrics.py
        
        self.id_model = np.random.randint(10000,99999)
        
        self.model = self.construct_network()
#        Calls the constructor of the architecture
        
        self.weight_factor = weight_factor
#        Weight factor, sometimes used in the dice loss
        
        return



    def __getstate__(self):
        state = self.__dict__
        del state['model']
        del state['metrics']
        del state['loss_function']
        del state['optimizer']
        state['training_results'] = state['training_results'].history
        return state




""" Definition of the functions that consitute the architecture : """

    @staticmethod
    def convolution_process(in_tensor, filters, dropout = False, **kwargs):
    """ Convolutions """
        c = tf.keras.layers.Conv2D(filters, (3, 3),
                                   padding = kwargs['padding'],
                                   activation=kwargs['activation'],
                                   kernel_initializer=kwargs['kernel_initializer'])(in_tensor)
        if dropout is not False:
            c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(filters, (3, 3),
                                   padding = kwargs['padding'],
                                   activation=kwargs['activation'],
                                   kernel_initializer=kwargs['kernel_initializer'])(c)
        return c



    @staticmethod
    def concat_process(tensor_1, tensor_2, n_filters):
    """ Concatenations """
        return tf.keras.layers.concatenate([
                    tf.keras.layers.Conv2DTranspose(n_filters, (2, 2),
                                            strides=(2, 2), padding='same')(tensor_1),
                    tensor_2])


""" Constructor of the network """

    def construct_network(self):
        [img_width, img_depth, img_channels] = self.img_dims
        inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
        intermediate_tensors_before_conv = [inputs]
        intermediate_tensors_after_conv = []

        # Encoding path
        for n_filters in self.architecture['encoding_path']:
            intermediate_tensors_after_conv.append(
                segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                                                           n_filters,
                                                                           **self.param_conv))
            intermediate_tensors_before_conv.append(
                tf.keras.layers.MaxPooling2D((2,2))(intermediate_tensors_after_conv[-1]))

        # Bottom
        intermediate_tensors_after_conv.append(
             segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                           self.architecture['bottom'],
                                           **self.param_conv))

        #Decoding_path
        for i in range(len(self.architecture['decoding_path'])):
            intermediate_tensors_before_conv.append(
                segmenter.concat_process(intermediate_tensors_after_conv[-1],
                                         intermediate_tensors_after_conv[self.depth - 2 - i],
                                         self.architecture['decoding_path'][i]))
            intermediate_tensors_after_conv.append(
                segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                              self.architecture['decoding_path'][i],
                                              **self.param_conv))

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(intermediate_tensors_after_conv[-1])



        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        model.summary()


        return model



    def train(self, X, Y, epochs, batch_size):
    """Training function that performs training of the network with training set X, ground truth mask Y """
        results = self.model.fit(X, Y, validation_split=0.1,
                                 batch_size=batch_size,
                                 epochs=epochs) #, callbacks=callbacks
        self.training_results = results
        training_curves(results)
        self.is_trained=True
        return results



    def evaluate(self,X,Y, display_prediction=False):
       """ Evaluate the network on set X with ground truth mask Y and display 5 random mask predictions"""
        if self.is_trained==False :
            warnings.warn("Networks Has not been trained")
        self.score = self.model.evaluate(X,Y)
        print(self.score)
        if display_prediction==True :
            n_data=X.shape[0]
            Random_indices= np.random.randint(low = 0, high= n_data,size =5)
            X2=X[Random_indices]
            Y2=Y[Random_indices]
            predict_example_and_plot(self.model,X2,Y2, size = self.img_dims[0])
        return
    
    
    def save(self):
    """ This funtion save and stores the results of the tests in pck files """

        file = open(os.path.join(model_dir,f'test_name/model_{self.id_model}.pck'), 'wb')

        pickle.dump(self, file)
        file.close()
        file = open(os.path.join(model_dir,'summary.txt'), 'a')
        file.write(f'model:{self.id_model} {self.architecture} {self.img_dims} {self.weight_factor}{self.score}\n')
        file.close()
        return




if __name__ == '__main__':
    img_dim = (96, 96, 1)
    test_split = 0.2
    n_images = 5000
        
    unet = UNET(img_dims=img_dim, loss_f=dice_loss_generator(factor), metrics=[dice_coef_eval], weight_factor=factor)
    unet.train(X_train, Y_train, epochs=20, batch_size=50)
    unet.evaluate(X_test, Y_test, display_prediction=True)
    unet.save()




