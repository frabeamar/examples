import tensorflow as tf
from keras.layers import Dense, Conv2D, InputLayer, Flatten, Conv2DTranspose,Reshape
from keras.models import Sequential
from keras.datasets import fashion_mnist
import keras
def build_autoencoder():

    encoder = Sequential(
        [ InputLayer((28, 28, 1)),
            Conv2D(32, 3, activation="relu", padding="same", strides=2),
            Conv2D(64, 3, activation="relu", padding="same", strides=2),  
            Flatten()  ,
            Dense(256, activation="relu")
        ]
    )
    
    decoder = Sequential(
        [
            InputLayer((2,)),
            Dense(7 * 7 * 64, activation="relu"),
            Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2),
            Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2),
            Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
        ]
    
    )
    print(encoder.summary())
    print(decoder.summary())
    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.xvar = Dense(2, activation="relu")
        self.xmean = Dense(2, activation="relu") 
        self.tot_loss = keras.metrics.Mean(name="loss"
        )
        self.reconstruct_loss = keras.metrics.Mean(name="reconstruct_loss")
        self.kl_loss = keras.metrics.Mean(name="kl_loss")

    
    @property
    def metrics(self):
        return [self.tot_loss, self.reconstruct_loss, self.kl_loss]


    def call(self, inputs):
        latent = self.encoder(inputs)
        mu = self.xmean(latent)
        logvar = self.xvar(latent)
        y = self.sample((mu, logvar))
        return self.decoder(y), mu, logvar, latent


    def sample(self, inputs):
        zMean, zLogVar = inputs
        batch = tf.shape(zMean)[0]
        dim = tf.shape(zMean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return zMean + tf.exp(0.5 * zLogVar) * epsilon


    def train_step(self, data):


        with tf.GradientTape() as tape:
            y, mu, logvar, z = self(data)
            loss = tot_loss(data, y, mu, logvar)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
        self.tot_loss.update_state(loss)
        self.reconstruct_loss.update_state(keras.losses.binary_crossentropy(data, y))
        self.kl_loss.update_state(divergence_loss(mu, logvar))
        return {"loss": self.tot_loss.result(), "reconstruct_loss": self.reconstruct_loss.result(), "kl_loss": self.kl_loss.result()}



def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.cast(X, tf.float32)
    X = tf.expand_dims(X, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(X/255.0).shuffle(1024).batch(32)
    return dataset


def divergence_loss(zMean, zLogVar):
  return tf.reduce_mean(
      tf.reduce_sum(
          -0.5 * (1 + zLogVar - tf.square(zMean) - tf.exp(zLogVar)),
          axis=1
      )
  )

def tot_loss(data, reconstructed, mu, logvar):
    recondstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstructed))
    kl_loss = divergence_loss(mu, logvar)
    return recondstruction_loss + kl_loss



if __name__ == "__main__":
    vae = VAE(*build_autoencoder())
    vae.compile(optimizer="adam", loss=tot_loss)
    dataset = load_dataset()
    vae.fit(dataset, epochs=10)


    
    print(vae.summary())
    