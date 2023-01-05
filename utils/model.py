from tensorflow import layers
import tensorflow as tf

def create_transfer_learning_model(base_model, 
                                   fine_tune=False,
                                   fine_tune_at=None,
                                   input_shape=(224,224,3)):
    
    # Freeze the base model
    base_model.Trainable = False
    
    if fine_tune :
        if not fine_tune_at:
            raise Exception("You should specify from which"+
                            " layer the model will be fine tuned"
                            )
        else:
            base_model.Trainable = True
            
            # Freeze the lowest layers 
            # and fine tune the top layers
            # starting from index "fine_tune_at"
            for layer in base_model.layers[:fine_tune_at]:
                layer.Trainable = False
                

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model