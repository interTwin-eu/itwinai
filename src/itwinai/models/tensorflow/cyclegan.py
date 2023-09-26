import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config


def residual_block(
        x,
        activation,
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        gamma_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
        x,
        filters,
        activation,
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        gamma_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        gamma_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02),
        use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


class Generator(keras.Model):
    def __init__(
            self,
            filters=64,
            num_downsampling_blocks=2,
            num_residual_blocks=9,
            num_upsample_blocks=2,
            gamma_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.02),
            input_img_size=(256, 256, 3)
    ):
        super().__init__()

        name = 'gen'

        self.filters = filters
        self.num_downsampling_blocks = num_downsampling_blocks
        self.num_residual_blocks = num_residual_blocks
        self.num_upsample_blocks = num_upsample_blocks
        self.gamma_initializer = gamma_initializer
        self.input_img_size = input_img_size

        img_input = layers.Input(shape=input_img_size,
                                 name=name + "_img_input")
        x = ReflectionPadding2D(padding=(3, 3))(img_input)
        x = layers.Conv2D(filters, (7, 7), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(
            x
        )
        x = tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_initializer)(x)
        x = layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = downsample(x, filters=filters,
                           activation=layers.Activation("relu"))

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = residual_block(x, activation=layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = upsample(x, filters, activation=layers.Activation("relu"))

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(3, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        self.model = keras.models.Model(img_input, x, name=name)

    def call(self, inputs, training=False):
        return self.model(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'num_downsampling_blocks': self.num_downsampling_blocks,
            'num_residual_blocks': self.num_residual_blocks,
            'num_upsample_blocks': self.num_upsample_blocks,
            'gamma_initializer': self.gamma_initializer,
            'input_img_size': self.input_img_size,
        })
        return config


class Discriminator(keras.Model):
    def __init__(
            self,
            filters=64,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.02),
            num_downsampling=3,
            input_img_size=(256, 256, 3)
    ):
        super().__init__()

        name = 'disc'
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.num_downsampling = num_downsampling
        self.input_img_size = input_img_size

        img_input = layers.Input(shape=input_img_size,
                                 name=name + "_img_input")
        x = layers.Conv2D(
            filters,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )(img_input)
        x = layers.LeakyReLU(0.2)(x)

        num_filters = filters
        for num_downsample_block in range(3):
            num_filters *= 2
            if num_downsample_block < 2:
                x = downsample(
                    x,
                    filters=num_filters,
                    activation=layers.LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(2, 2),
                )
            else:
                x = downsample(
                    x,
                    filters=num_filters,
                    activation=layers.LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(1, 1),
                )

        x = layers.Conv2D(
            1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
        )(x)
        self.model = keras.models.Model(inputs=img_input, outputs=x, name=name)

    def call(self, inputs, training=False):
        return self.model(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_initializer': self.kernel_initializer,
            'num_downsampling': self.num_downsampling,
            'input_img_size': self.input_img_size,
        })
        return config


class CycleGAN(keras.Model):
    def __init__(
            self,
            generator_G: keras.Model,
            generator_F: keras.Model,
            discriminator_X: keras.Model,
            discriminator_Y: keras.Model,
            lambda_cycle=10.0,
            lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, config: dict):
        super().compile()
        self.gen_G_optimizer = config['gen_G_optimizer']
        self.gen_F_optimizer = config['gen_F_optimizer']
        self.disc_X_optimizer = config['disc_X_optimizer']
        self.disc_Y_optimizer = config['disc_Y_optimizer']

        # TODO: Define losses in config file
        # Loss function for evaluating adversarial loss
        adv_loss_fn = keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM)

        # Define the loss function for the generators
        def generator_loss_fn(fake):
            fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
            return fake_loss

        # Define the loss function for the discriminators
        def discriminator_loss_fn(real, fake):
            real_loss = adv_loss_fn(tf.ones_like(real), real)
            fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
            return (real_loss + fake_loss) * 0.5

        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

        self.cycle_loss_fn = keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM)
        self.identity_loss_fn = keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM)

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adversarial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(
                real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(
                real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(
            disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(
            disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

    def test_step(self, inputs):
        real_x, real_y = inputs

        # Horse to fake zebra
        fake_y = self.gen_G(real_x, training=False)
        # Zebra to fake horse -> y2x
        fake_x = self.gen_F(real_y, training=False)

        # Cycle (Horse to fake zebra to fake horse): x -> y -> x
        cycled_x = self.gen_F(fake_y, training=False)
        # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
        cycled_y = self.gen_G(fake_x, training=False)

        # Identity mapping
        same_x = self.gen_F(real_x, training=False)
        same_y = self.gen_G(real_y, training=False)

        # Discriminator output
        disc_real_x = self.disc_X(real_x, training=False)
        disc_fake_x = self.disc_X(fake_x, training=False)

        disc_real_y = self.disc_Y(real_y, training=False)
        disc_fake_y = self.disc_Y(fake_y, training=False)

        # Generator adversarial loss
        gen_G_loss = self.generator_loss_fn(disc_fake_y)
        gen_F_loss = self.generator_loss_fn(disc_fake_x)

        # Generator cycle loss
        cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
        cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

        # Generator identity loss
        id_loss_G = (
            self.identity_loss_fn(real_y, same_y)
            * self.lambda_cycle
            * self.lambda_identity
        )
        id_loss_F = (
            self.identity_loss_fn(real_x, same_x)
            * self.lambda_cycle
            * self.lambda_identity
        )

        # Total generator loss
        total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
        total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

        # Discriminator loss
        disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
        disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'generator_G': self.gen_G,
            'generator_F': self.gen_F,
            'discriminator_X': self.disc_X,
            'discriminator_Y': self.disc_Y,
            'lambda_cycle': self.lambda_cycle,
            'lambda_identity': self.lambda_identity,
        })
        return config
