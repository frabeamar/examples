from tensorflow.keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Reshape, Conv2D, Flatten, Dropout
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

NOISE_DIM = 128
BATCH_SIZE = 32


@tf.function
def trainDstep(
    generator: keras.Model,
    discriminator: keras.Model,
    lossfn: keras.losses.Loss,
    optimizerD: keras.optimizers.Adam,
    dAccMetric: keras.metrics.Metric,
    data,
):
    batch_size = data.shape[0]
    noise = tf.random.normal((batch_size, NOISE_DIM))
    y_true = tf.concat(
        [
            tf.ones((batch_size, 1), dtype=tf.int32),
            tf.zeros((batch_size, 1), dtype=tf.int32),
        ],
        axis=0,
    )
    with tf.GradientTape() as tape:
        fake = generator(noise)
        x = tf.concat([data, fake], axis=0)
        y_pred = discriminator(x)
        loss = lossfn(y_true, y_pred)
        grads = tape.gradient(loss, discriminator.trainable_weights)
        optimizerD.apply_gradients(zip(grads, discriminator.trainable_weights))
        dAccMetric.update_state(y_true, y_pred)
        return {"d_loss": loss, "d_acc": dAccMetric.result()}


@tf.function
def trainGstep(
    discriminator: keras.Model,
    generator: keras.Model,
    lossfn: keras.losses.Loss,
    optimizerG: keras.optimizers.Adam,
    Gmetrics: keras.metrics.Metric,
):
    # train the generator
    # want to maximize the probablity of generating true labels
    y_true = tf.ones((BATCH_SIZE, 1))

    with tf.GradientTape() as tape:
        noise = tf.random.normal((BATCH_SIZE, NOISE_DIM))
        y_pred = discriminator(generator(noise))
        loss = lossfn(y_true, y_pred)
        grads = tape.gradient(loss, generator.trainable_weights)
        optimizerG.apply_gradients(zip(grads, generator.trainable_weights))
        Gmetrics.update_state(y_true, y_pred)

    return {"g_loss": loss, "g_acc": Gmetrics.result()}


def plot_images(generator: keras.Model):
    noise = tf.random.normal(shape=(81, NOISE_DIM))
    images = generator(noise)
    plt.figure(figsize=(9, 9))
    for i, image in enumerate(images):
        plt.subplot(9, 9, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.show()


def build_generator():
    model = Sequential(
        [
            keras.layers.InputLayer(input_shape=(NOISE_DIM,)),
            Dense(7 * 7 * 256),
            Reshape(target_shape=(7, 7, 256)),
            Conv2DTranspose(256, 3, activation="leaky_relu", strides=2, padding="same"),
            Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same"),
            Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ]
    )
    print(model.summary())
    return model


def build_descriminator():
    model = Sequential(
        [
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            Conv2D(256, 3, strides=2, activation="relu", padding="same"),
            Conv2D(128, 3, strides=2, activation="relu", padding="same"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    print(model.summary())
    return model


def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    generator = build_generator()
    descriminator = build_descriminator()
    optimizerG = keras.optimizers.Adam(0.0001)
    optimizerD = keras.optimizers.Adam(0.0003)
    lossfn = keras.losses.BinaryCrossentropy()
    metricsG = keras.metrics.BinaryAccuracy()
    metricsD = keras.metrics.BinaryAccuracy()

    dataset = tf.concat([x_train, x_test], axis=0)
    dataset = tf.expand_dims(dataset, axis=-1)
    dataset = tf.cast(dataset, tf.float32)
    dataset = dataset / 255.0

    dataset = (
        tf.data.Dataset.from_tensor_slices(dataset)
        .shuffle(buffer_size=1024)
        .batch(BATCH_SIZE)
    )
    for epoch in range(30):
        dsum = 0
        dacc = 0
        gsum = 0
        gacc = 0
        pbar = tqdm.tqdm(enumerate(dataset), desc="Training")
        for i, data in pbar:
            dLoss = trainDstep(
                generator, descriminator, lossfn, optimizerD, metricsD, data
            )
            gLoss = trainGstep(descriminator, generator, lossfn, optimizerG, metricsG)
            dsum = dsum + dLoss["d_loss"]
            dacc = dacc + dLoss["d_acc"]
            gsum = gsum + gLoss["g_loss"]
            gacc = gacc + gLoss["g_acc"]
            pbar.set_postfix({"dloss": f"{dsum / i:.3f}", "gloss": f"{gsum / i:.3f}"})

        generator.save("generator.keras")
        descriminator.save("descriminator.keras")
        plot_images(generator)


if __name__ == "__main__":
    main()
