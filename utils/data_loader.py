import numpy as np
import tensorflow as tf
import os
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras import layers

HOME_DIR = os.path.dirname(os.path.dirname(__file__))
CONF_PATH = os.path.join(HOME_DIR,"labels.csv")

class DataLoader:
    def __init__(self,path=CONF_PATH,
                    image_size=224, 
                    shuffle=True, 
                    augment=True,
                    batch_size=32,
                    val_size = 0.2,
                    random_state=42,
                    include_cell_type=False) -> None:
        
        # The model can support 
        # images of size 192 or 224 only
        if image_size!= 224 and image_size!= 192:
            raise Exception("Image size can only be 224 or 192")
        
        # Wether to use the cell type
        # as a feature for the model
        self.include_cell_type = include_cell_type
        
        self.augment = augment
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state
        self.shuffle  = shuffle
        self.path = path
        self.image_size = image_size
        
        
    def load_dataset(self):
        # Load dataset    
        images, defect_probas, cell_types = load_dataset(fname=self.path)
        
        # preprocess dataset
        self.X1, self.X2, self.Y = self._preprocess_data(images=images,
                                          defect_proba=defect_probas,
                                          cell_type=cell_types)
        
        # Stratify split dataset
        # to train and validation 

        X1_train,X1_val,Y_train,Y_val = train_test_split(self.X1,self.Y,
                                                         test_size=self.val_size,
                                                         random_state=self.random_state,
                                                         stratify=self.Y)
        
        X2_train,X2_val,Y_train,Y_val = train_test_split(self.X2,self.Y,
                                                           test_size=self.val_size,
                                                           random_state=self.random_state,
                                                           stratify=self.Y)
        
        # create tensorflow datasets
        if not self.include_cell_type:
            train_dataset, val_dataset = self._create_tf_dataset(x_train=X1_train,
                                                                 x_val=X1_val,
                                                                 y_train=Y_train,
                                                                 y_val=Y_val)
            train_batches = self.batch_data(train_dataset,
                                            augment=self.augment,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            val_batches = self.batch_data(val_dataset,
                                            augment=False,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            
            return train_batches, val_batches
        
        else:
            train1_dataset,val1_dataset, train2_dataset, val2_dataset = self._create_tf_dataset(x1_train=X1_train,
                                                                 x1_val=X1_val,
                                                                 x2_train=X2_train,
                                                                 x2_val=X2_val,
                                                                 y_train=Y_train,
                                                                 y_val=Y_val)
            train1_batches = self.batch_data(train1_dataset,
                                            augment=self.augment,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            val1_batches = self.batch_data(val1_dataset,
                                            augment=False,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            # TODO verify this feature 
            # since we can't augment the data
            # and the function applies image rescaling
            train2_batches = self.batch_data(train2_dataset,
                                            augment=False,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            val2_batches = self.batch_data(val2_dataset,
                                            augment=False,
                                            shuffle=self.shuffle,
                                            batch_size=self.batch_size,
                                            image_size=self.image_size)
            
            return train1_batches, val1_batches, train2_batches, val2_batches
        
        
    def batch_data(self,ds,
                   augment=True,
                   shuffle=True,
                   batch_size=32,
                   image_size=224,
                   buffer_size=1000
                   ):
        
        AUTOTUNE = tf.data.AUTOTUNE

        IMG_SIZE = image_size
        
        # Rescaling the images
        resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255)
        ])

        # Defining data augmentation technique
        data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        ])


        # Batch all datasets.
        ds = ds.batch(batch_size)
        
        # Resize and rescale all datasets.
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                    num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size)


        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=AUTOTUNE)

        
        # Use buffered prefetching on all datasets.
        ds = ds.cache()
        return ds.prefetch(buffer_size=AUTOTUNE)
    
    def _preprocess_data(self,images,defect_proba,cell_type):
        
        # Convert the probabilities to classes
        Y = defect_proba.copy()
        Y[Y >= 0.5] = 1. # the cell is defective
        Y[Y < 0.5] = 0. # the cell is not defective

        # Convert grayscale to rgb
        X1 = self._grayscale_to_rgb(images)
        
        # convert cell type to numerical data
        # in case it is used as a second feature
        X2 = cell_type.copy()
        if self.include_cell_type:
            X2[X2 == "mono"] = 0
            X2[X2 == "poly"] = 1
        
        return X1,X2,Y
    
    def _grayscale_to_rgb(self,images):
        # Convert grayscale image to rgb
        # by repeating the grayscale image
        # over the three channels
        # This is needed to adapt the images
        # to the inputs of the transfer learning model
        
        # images.shape = (batch_size,x,y)
        # rgb_imgs.shape = (batch_size,x,y,3)
        rgb_imgs = np.repeat(images[..., np.newaxis],3,-1)
        return rgb_imgs
    
    def _create_tf_dataset(self, **kwargs):
        
        try:
            if "x1_train" in kwargs.keys():
                
                X1_train = kwargs["x1_train"]
                X2_train = kwargs["x2_train"]
                
                X1_val = kwargs["x1_val"]
                X2_val = kwargs["x2_val"]
                
                Y_train = kwargs["y_train"]
                Y_val = kwargs["y_val"]
                
                # Creating tensorflow datasets
                # for the two features : images + cell type
                train1_dataset = Dataset.from_tensor_slices((X1_train,Y_train))
                val1_dataset = Dataset.from_tensor_slices((X1_val,Y_val))
                
                train2_dataset = Dataset.from_tensor_slices((X2_train,Y_train))
                val2_dataset = Dataset.from_tensor_slices((X2_val,Y_val))
                
                return train1_dataset,val1_dataset, train2_dataset, val2_dataset
            
            elif "x_train" in kwargs.keys():
                X_train = kwargs["x_train"]
                X_val = kwargs["x_val"]
                
                
                Y_train = kwargs["y_train"]
                Y_val = kwargs["y_val"]
                
                # Creating tensorflow datasets
                # for images
                train_dataset = Dataset.from_tensor_slices((X_train,Y_train))
                val_dataset = Dataset.from_tensor_slices((X_val,Y_val))
                
                return train_dataset, val_dataset
            else : 
                raise KeyError("The only key allowed are:"+
                           " (x_train,y_train,x_val,y_val)"+
                           " or (x1_train, x2_train,y_train,"+
                           " x1_val, x2_val, y_val)")
                   
        except KeyError:
            raise KeyError("The only key allowed are:"+
                           " (x_train,y_train,x_val,y_val)"+
                           " or (x1_train, x2_train,y_train,"+
                           " x1_val, x2_val, y_val)")
            
       