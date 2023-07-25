import tensorflow as tf
from .architectures.vgg import custom_VGG_V1, custom_VGG_V2, custom_VGG_V3, ModelV5
from ..macros import Network



def get_network_config(network, **kwargs):
    # choose the network configuration based on the passed network type
    if network == Network.VGG_V1.value:
        print('Using custom VGG V1')
        model = custom_VGG_V1(patch_size=kwargs['patch_size'], channels=kwargs['channels'], activation=kwargs['activation'], regularizer=kwargs['regularizer'])
    
    elif network == Network.VGG_V2.value:
        print('Using custom VGG V2')
        model = custom_VGG_V2(patch_size=kwargs['patch_size'], channels=kwargs['channels'], activation=kwargs['activation'], regularizer=kwargs['regularizer'])
    
    elif network == Network.VGG_V3.value:
        print('Using custom VGG V3')
        model = custom_VGG_V3(patch_size=kwargs['patch_size'], channels=kwargs['channels'], activation=kwargs['activation'], regularizer=kwargs['regularizer'])

    elif network == Network.MODEL_V5.value:
        print('Using Model V5')
        model = ModelV5(patch_size=kwargs['patch_size'], channels=kwargs['channels'], last_activation=kwargs['activation'], kernel_size=kwargs['kernel_size'])

    return model



def load_model(model_fpath):
    """
    Loads a keras model from a file, recognizing whether or not it is a weight file or a model file.

    """
    model_fname = model_fpath.split('/')[-1]
    if 'model' in model_fname:
        try:
            model = tf.keras.models.load_model(model_fpath)
        except Exception as e:
            print(f'Cannot load model. Caused by error: {e}')
    elif 'weight' in model_fname:
        model.compile()
        model.built = True
        model.load_weights(model_fpath)
    return model