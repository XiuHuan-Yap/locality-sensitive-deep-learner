import numpy as np
import tensorflow as tf
from algorithms import attention_model_LeakyReLU


class AttentionModelExp(object):
    def __init__(self,
                 X, y, X_val, y_val, Fweights=None, Fweights_val=None, 
                 model_type="Attention_w_FW",
                 n_batch=256,
                 model_param_dict={},
                 model_hyperparam_dict={},
                 epochs=1000,
                 callbacks=[],
                 verbose=0,
                 random_seed=123):

        self.model_type=model_type
        self.n_batch=n_batch
        self.model_param_dict=model_param_dict
        self.model_hyperparam_dict=model_hyperparam_dict
        self.epochs=epochs
        self.verbose=verbose
        self.random_seed=random_seed
        self.callbacks=callbacks

        self._check_inputs(X.shape, X_val.shape, Fweights, Fweights_val)
        self._assemble_dataset(X, y, X_val, y_val, Fweights, Fweights_val)
        
    def _check_inputs(self, X_shape, X_val_shape, Fweights, Fweights_val):
        if self.model_type=="Attention_w_FW":
            assert np.all([Fweights is not None, Fweights_val is not None]), "Parameter `Fweights` or `Fweights_val` should not be None"
            assert np.all([Fweights.shape==X_shape, Fweights_val.shape==X_val_shape]), "Parameter `Fweights` and `Fweights_val` should have same shape as parameter `X` and `X_val` respectively"
        elif (self.model_type!="Attention_wo_FW") & (self.model_type!="Dense"):
            assert self.model_type=="Dense", "Parameter `model` should be 'Attention_w_FW', 'Attention_wo_FW' or 'Dense'"
        
    def _assemble_dataset(self, X, y, X_val, y_val, Fweights, Fweights_test):
        if self.model_type=="Attention_w_FW":
            train_ds=tf.data.Dataset.from_tensor_slices(((X, Fweights), y))
            val_ds=tf.data.Dataset.from_tensor_slices(((X_val, Fweights_test), y_val))
            self.Fweights=Fweights
        else:
            train_ds=tf.data.Dataset.from_tensor_slices((X,y))
            val_ds=tf.data.Dataset.from_tensor_slices((X_val, y_val))
            
        self.train_ds=train_ds.batch(self.n_batch)
        self.val_ds=val_ds.batch(self.n_batch)
        
    def fit(self):
        np.random.seed(self.random_seed)        
        #Construct model
        
        if self.model_type=="Attention_w_FW":
            model=attention_model_LeakyReLU.AttentionModelwFeatWeights(
                **self.model_param_dict
                )
            
        elif self.model_type=="Attention_wo_FW":
            model=attention_model_LeakyReLU.AttentionModel(**self.model_param_dict)
            
        elif model_type=="Dense":
            model=attention_model_LeakyReLU.DenseComparison(**self.model_param_dict)
        
        model.compile(**self.model_hyperparam_dict)
        
        if self.model_type=="Attention_w_FW":
            model.fit(self.train_ds, epochs=1, verbose=0)
            #Randomly set feature weights
            sampled_Fweights=self.Fweights[np.random.choice(range(len(self.Fweights)), 
                                                            model.n_attention)]
            for i in range(model.n_attention):
                weights=model.attentions.attention_layers[i].get_weights()
                weights[0]=np.reshape(sampled_Fweights[i], (1,self.Fweights.shape[1]))
                model.attentions.attention_layers[i].set_weights(weights)
            del self.Fweights
            
        model.fit(self.train_ds, 
                  epochs=self.epochs, 
                  validation_data=self.val_ds,
                  verbose=self.verbose,
                  callbacks=self.callbacks
                 )
        
        return model