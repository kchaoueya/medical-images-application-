"""U-Net du TP4 (identique à `simple_unet_model` du notebook) + fonctions de perte."""


def build_unet(h=256, w=256, c=1, dropout_scale=1.0):
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout)
    from tensorflow.keras.models import Model
    conv = dict(kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")
    d = lambda r: Dropout(min(r * dropout_scale, 0.6))  # noqa: E731

    inputs = Input((h, w, c))
    # --- encodeur : filtres 16, 32, 64, 128 puis goulot à 256 (sans MaxPooling)
    c1 = Conv2D(16, **conv)(inputs); c1 = d(.1)(c1); c1 = Conv2D(16, **conv)(c1); p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, **conv)(p1); c2 = d(.1)(c2); c2 = Conv2D(32, **conv)(c2); p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, **conv)(p2); c3 = d(.2)(c3); c3 = Conv2D(64, **conv)(c3); p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, **conv)(p3); c4 = d(.2)(c4); c4 = Conv2D(128, **conv)(c4); p4 = MaxPooling2D((2, 2))(c4)
    c5 = Conv2D(256, **conv)(p4); c5 = d(.3)(c5); c5 = Conv2D(256, **conv)(c5)
    # --- décodeur : Conv2DTranspose -> concaténation (skip) -> Conv -> Dropout -> Conv
    u6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5), c4])
    c6 = Conv2D(128, **conv)(u6); c6 = d(.2)(c6); c6 = Conv2D(128, **conv)(c6)
    u7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6), c3])
    c7 = Conv2D(64, **conv)(u7); c7 = d(.2)(c7); c7 = Conv2D(64, **conv)(c7)
    u8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7), c2])
    c8 = Conv2D(32, **conv)(u8); c8 = d(.1)(c8); c8 = Conv2D(32, **conv)(c8)
    u9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8), c1])
    c9 = Conv2D(16, **conv)(u9); c9 = d(.1)(c9); c9 = Conv2D(16, **conv)(c9)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return Model(inputs=[inputs], outputs=[outputs])


def get_loss(name):
    """'bce' (TP4), 'dice' ou 'bce_dice' (cellule « not sure » du notebook) : utiles quand la région est rare (≈9 % des pixels)."""
    import tensorflow as tf
    from tensorflow.keras import backend as K

    def dice_loss(y_true, y_pred, smooth=1.0):
        a, b = K.flatten(y_true), K.flatten(y_pred)
        return 1 - (2 * K.sum(a * b) + smooth) / (K.sum(a) + K.sum(b) + smooth)

    def bce_dice(y_true, y_pred):
        return K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred)) + dice_loss(y_true, y_pred)

    return {"bce": "binary_crossentropy", "dice": dice_loss, "bce_dice": bce_dice}[name]
