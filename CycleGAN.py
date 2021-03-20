from tensorflow import keras
import tensorflow as tf

class CycleGan(keras.Model):
    def __init__(
        self,
        c_generator,
        s_generator,
        c_discriminator,
        s_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.c_gen = c_generator
        self.s_gen = s_generator

        
        self.c_disc = c_discriminator
        self.s_disc = s_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        c_gen_optimizer,
        s_gen_optimizer,
        c_disc_optimizer,
        s_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.c_gen_optimizer = c_gen_optimizer
        self.s_gen_optimizer = s_gen_optimizer
        self.c_disc_optimizer = c_disc_optimizer
        self.s_disc_optimizer = s_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_s, real_c = batch_data
        
        with tf.GradientTape(persistent=True) as tape:

            fake_s = self.s_gen(real_c, training=True)
            cycled_c = self.c_gen(fake_s, training=True)

            fake_c = self.c_gen(real_s, training=True)
            cycled_s = self.s_gen(fake_c, training=True)

            # generating itself
            same_s = self.s_gen(real_s, training=True)
            same_c = self.c_gen(real_c, training=True)

            # discriminator used to check, inputing real images
            disc_real_s = self.s_disc(real_s, training=True)
            disc_real_c = self.c_disc(real_c, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_s = self.s_disc(fake_s, training=True)
            disc_fake_c = self.c_disc(fake_c, training=True)

            # evaluates generator loss
            s_gen_loss = self.gen_loss_fn(disc_fake_s)
            c_gen_loss = self.gen_loss_fn(disc_fake_c)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_s, 
                                                  cycled_s, 
                                                  self.lambda_cycle) + self.cycle_loss_fn(real_c, 
                                                                     cycled_c, 
                                                                     self.lambda_cycle)

            # evaluates total generator loss
            total_s_gen_loss = s_gen_loss + total_cycle_loss + self.identity_loss_fn(real_s, 
                                                                                     same_s, 
                                                                                     self.lambda_cycle)
            total_c_gen_loss = c_gen_loss + total_cycle_loss + self.identity_loss_fn(real_c, 
                                                                                     same_c, 
                                                                                     self.lambda_cycle)

            # evaluates discriminator loss
            s_disc_loss = self.disc_loss_fn(disc_real_s, disc_fake_s)
            c_disc_loss = self.disc_loss_fn(disc_real_c, disc_fake_c)


        # Calculate the gradients for generator and discriminator
        s_generator_gradients = tape.gradient(total_s_gen_loss,
                                                  self.s_gen.trainable_variables)
        c_generator_gradients = tape.gradient(total_c_gen_loss,
                                                  self.c_gen.trainable_variables)

        s_discriminator_gradients = tape.gradient(s_disc_loss,
                                                  self.s_disc.trainable_variables)
        c_discriminator_gradients = tape.gradient(c_disc_loss,
                                                  self.c_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.s_gen_optimizer.apply_gradients(zip(s_generator_gradients,
                                                 self.s_gen.trainable_variables))

        self.c_gen_optimizer.apply_gradients(zip(c_generator_gradients,
                                                 self.c_gen.trainable_variables))
        
        self.s_disc_optimizer.apply_gradients(zip(s_discriminator_gradients,
                                                  self.s_disc.trainable_variables))

        self.c_disc_optimizer.apply_gradients(zip(c_discriminator_gradients,
                                                  self.c_disc.trainable_variables))
        
        return {
            "DomA_gen_loss": total_s_gen_loss,
            "DomB_gen_loss": total_c_gen_loss,
            "DomA_disc_loss": s_disc_loss,
            "DomB_disc_loss": c_disc_loss
        }
    
    def call(self, x):
        pass