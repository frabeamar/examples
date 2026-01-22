import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.metrics import Precision, Recall
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
# has not the same accuracy as the pytorch version
# -------------------------------------------------
# Dataset utilities
# -------------------------------------------------

def load_cifar10(batch_size=32):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # .map(resize, num_parallel_calls=tf.data.AUTOTUNE) # Add resizing here
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


# -------------------------------------------------
# Data augmentation
# -------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.05),
        layers.GaussianNoise(0.05),
  ],
    name="augmentation",
)


# -------------------------------------------------
# Model
# -------------------------------------------------
def build_model(model_type: str, num_classes: int = 10):
    backbone_map = {
        "efficientnet_b3": keras.applications.EfficientNetB3,
        "resnet50v2": keras.applications.ResNet50V2,
        "resnet101v2": keras.applications.ResNet101V2,
        "mobilenetv2": keras.applications.MobileNetV2,
    }

    backbone_fn = backbone_map[model_type]

    backbone = backbone_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
        pooling="avg",
    )
    backbone.trainable = True

    inputs = keras.Input(shape=(32, 32, 1))
    x = data_augmentation(inputs)
    x = backbone(x)
    outputs = layers.Dense(num_classes)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),             
                    # Precision(name='precision'),
                    # Recall(name='recall')
                ],
    )

    return model


# -------------------------------------------------
# Training
# -------------------------------------------------
def train():
    train_ds, test_ds = load_cifar10()

    model_types = [
        # "efficientnet_b3",
        # "resnet50v2",
        # "resnet101v2",
        "mobilenetv2",
    ]

    for m in model_types:
        logdir = f"./logs/{m}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tb_cb = keras.callbacks.TensorBoard(log_dir=logdir)

        model = build_model(m)

        model.fit(
            train_ds,
            epochs=10,
            validation_data=test_ds,
            callbacks=[tb_cb],

        )
        breakpoint()
        loss, accuracy = model.evaluate(test_ds)

if __name__ == "__main__":
    train()
