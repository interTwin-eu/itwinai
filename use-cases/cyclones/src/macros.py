import tensorflow as tf
from enum import Enum

# default loss configuration reduction
REDUCTION = tf.keras.losses.Reduction.NONE

#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                      #
#                           NetCFD Variables                           #
#                                                                      #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TUTTE LE VARIABILI DISPONIBILI
ALL_DRIVER_VARS = ['fg10', 'i10fg', 'msl', 'sst', 't_500', 't_300', 'vo_850']
ALL_COORDINATE_VARS = ['real_cyclone',
                       'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
CYCLONE_VAR = 'patch_cyclone'
MASK_VAR = 'cyclone_mask'

# ESPERIMENTI TIPO 1
# variabili per la prima parte degli esperimenti (regressione per trovare
# coordinate row-col intra-patch)
EXPERIMENT_1 = {
    'DRV_VARS_1': ['fg10', 'msl', 't_500', 't_300'],
    'COO_VARS_1': ['patch_cyclone'],
    'MSK_VAR_1': None
}

# dataset parameters
PATCH_SIZE = 40
SHAPE = (PATCH_SIZE, PATCH_SIZE)

TEST_YEARS = []  # for test purposes
TRAINVAL_YEARS = [2000, 2001, 2002]  # for test purposes


#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                      #
#                            Enumerations                              #
#                                                                      #
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# descrive il tipo di patch che bisogna prendere
class PatchType(Enum):
    ALLADJACENT = 'alladjacent'
    CYCLONE = 'cyclone'
    NEAREST = 'nearest'
    RANDOM = 'random'
    NOCYCLONE = 'nocyclone'

# descrive il tipo di augmentation che deve essere effettuata


class AugmentationType(Enum):
    ALL_PATCHES = 'all_patches'
    ONLY_TCS = 'only_tcs'

# descrive il nome del modello di rete neurale da utilizzare


class Network(Enum):
    VGG_V1 = 'vgg_v1'       # map-to-coord
    VGG_V2 = 'vgg_v2'       # map-to-coord
    VGG_V3 = 'vgg_v3'       # map-to-coord
    MODEL_V5 = 'model_v5'   # map-to-coord

# ritorna nome della loss utilizzata in fase di training


class Losses(Enum):
    # Mean Absolute Error
    MAE = ('mae', 'mae')
    # Mean Squared Error
    MSE = ('mse', 'mse')
    # No specified loss
    NONE = ('none', None)

# descrive la forza della regolarizzazione


class RegularizationStrength(Enum):
    WEAK = ('weak', tf.keras.regularizers.l1_l2(
        l1=0.0, l2=0.0001))  # l1=0 - l2=0.0001
    MEDIUM = ('medium', tf.keras.regularizers.l1_l2(
        l1=0.0001, l2=0.0001))  # l1=0.0001 - l2=0.0001
    STRONG = ('strong', tf.keras.regularizers.l1_l2(
        l1=0.001, l2=0.001))  # l1=0.001 - l2=0.001
    VERY_STRONG = ('very_strong', tf.keras.regularizers.l1_l2(
        l1=0.01, l2=0.01))  # l1=0.01 - l2=0.01
    NONE = ('none', None)   # no regularization

# descrive l'attivazione dell'ultimo layer del modello


class Activation(Enum):
    RELU = 'relu'
    LINEAR = 'linear'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

# label assegnata ad un ciclone assente


class LabelNoCyclone(Enum):
    ZERO_3 = -0.3
    ONE = -1.0
    NONE = None
