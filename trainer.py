from DatasetLoader import  GANDataGenerator,  GANDataGeneratorXY
from Datasetlabels import *
import tensorflow as tf
from losses import *
from generator import Generator
from discriminator import Discriminator
from CycleGAN import CycleGan

tf.compat.v1.reset_default_graph()
def train(config):
    train_gen=None
    if config.subject==0:
        # dataset-loder used in case of sketch to colorize image
        train_gen = GANDataGenerator(train_imgs,
                                     config.dataset,
                                     config.batch,
                                     dim = (config.height, config.width)
                                     )
    else:
        # dataset-loader used in case of gender-bender and glass to no-glass
        train_gen = GANDataGeneratorXY(domainB,
                                       domainG,
                                       config.dataset,
                                       config.batch,
                                       dim = (config.height, config.width)
                                       )

    c_generator = Generator(config.height,config.width, config.alpha) # transforms Domain-1 to Domain-2
    s_generator = Generator(config.height,config.width, config.alpha) # transforms Domain-2 to be more like Domain-1

    c_discriminator = Discriminator(config.height,config.width, config.alpha) # differentiates real Domain-1 and generated Domain-1
    s_discriminator = Discriminator(config.height,config.width, config.alpha) # differentiates real Domain-2 and generated Domain-2

    c_generator_optimizer = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1, beta_2=config.beta2, 
                                        epsilon=config.epsilon)
    s_generator_optimizer = tf.keras.optimizers.Adam(config.gen_lr, beta_1=config.beta1, beta_2=config.beta2, 
                                        epsilon=config.epsilon)

    c_discriminator_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1, beta_2=config.beta2, 
                                        epsilon=config.epsilon)
    s_discriminator_optimizer = tf.keras.optimizers.Adam(config.dis_lr, beta_1=config.beta1, beta_2=config.beta2, 
                                        epsilon=config.epsilon)

    gan_model = CycleGan(
            c_generator, s_generator, c_discriminator, 
            s_discriminator
        )

    gan_model.compile(
            c_gen_optimizer = c_generator_optimizer,
            s_gen_optimizer = s_generator_optimizer,
            c_disc_optimizer = c_discriminator_optimizer,
            s_disc_optimizer = s_discriminator_optimizer,
            gen_loss_fn = generator_loss,
            disc_loss_fn = discriminator_loss,
            cycle_loss_fn = calc_cycle_loss,
            identity_loss_fn = identity_loss
        )

    gan_model.fit(
            train_gen,
            epochs=config.epoch
        )
    return c_generator

