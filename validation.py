from tensorflow import keras
import matplotlib.pyplot as plt
from generator import Generator

def testModel(
    test_input=None, 
    target=None, 
    config=None):
    
    Eval_generator = Generator(config.height,config.width, config.alpha)
    Eval_generator.load_weights(config.model_save_path+'model_weights')

    prediction =  Eval_generator(test_input, training=False)
    if config.subject==0:
        display_list = [test_input[0], target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow((display_list[i]+ 1) / 2.0)
            plt.axis('off')
    
    if config.subject==1:
        title = ['Input Image', 'Predicted Image']
        display_list = [test_input[0], prediction[0]]

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow((display_list[i]+ 1) / 2.0)
            plt.axis('off')

    plt.savefig("Gan_Output_"+config.validate)