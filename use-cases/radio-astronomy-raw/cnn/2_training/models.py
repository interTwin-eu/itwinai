from tensorflow.keras import layers, models


def model_PROT_LE_1_230908_0(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(4, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model



def model_PROT_LE_1_230908_1(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(8, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model


def model_PROT_LE_1_230908_2(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model


def model_PROT_LE_1_230909_0(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(4, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (5, 5), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model


def model_PROT_LE_1_230909_1(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(8, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (5, 5), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model


def model_PROT_LE_1_230909_2(resol):

    model = models.Sequential()

    model.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(resol, resol, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (5, 5), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    
    return model


models_htable = {
	'PROT_LE_1_230908_0': model_PROT_LE_1_230908_0,
	'PROT_LE_1_230908_1': model_PROT_LE_1_230908_1,
	'PROT_LE_1_230908_2': model_PROT_LE_1_230908_2,
	'PROT_LE_1_230909_0': model_PROT_LE_1_230908_0,
	'PROT_LE_1_230909_1': model_PROT_LE_1_230908_1,
	'PROT_LE_1_230909_2': model_PROT_LE_1_230908_2,
}
