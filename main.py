from trainer import train
from validation import testModel
from DatasetLoader import  GANDataGenerator,  GANDataGeneratorXY
import numpy as np
from Datasetlabels import *

from config import get_parameters
def main(config):
    if config.train:
        '''
           Train the CycleGAN Model
        '''
        model = train(config)
        # saving model weights
        model.save_weights(config.model_save_path+'model_weights')

    else:
        '''
           Validate the CycleGAN Model
        '''
        validation_image_path = np.array([config.validate])
        if config.subject==0:
            # dataset-loder used in case of sketch to colorize image
            loader = GANDataGenerator(validation_image_path,
                                        config.dataset,
                                        1,
                                        dim = (config.height, config.width)
                                        )
        else:
            # dataset-loader used in case of gender-bender and glass to no-glass
            loader = GANDataGeneratorXY(validation_image_path,
                                        validation_image_path,
                                        config.dataset,
                                        1,
                                        dim = (config.height, config.width)
                                        )
        source, destination =next(iter(loader))
        testModel(source, destination, config)

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)