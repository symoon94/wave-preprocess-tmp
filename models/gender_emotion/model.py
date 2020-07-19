import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

INIT_LR = 1e-3
EPOCHS = 1000

height = 256
width = 250
channel = 1

class AudioNet:
    def __init__(self, height, width, channel=1, num_emotions=2, num_genders=2,
		emo_act="sigmoid", gen_act="softmax"):
        
        self.height = height
        self.width = width
        self.channel = channel
        self.num_emotions = num_emotions
        self.num_genders = num_genders
        self.emo_act = emo_act
        self.gen_act = gen_act

    def build_emotion_branch(self, inputs):
        x = layers.Dense(256)(inputs)
        x = layers.Dense(128)(x)
        x = layers.Dense(32)(x)
        x = layers.Dense(self.num_emotions, activation=self.emo_act, name="emotion_output")(x)
        return x

    def build_gender_branch(self, inputs):
        x = layers.Dense(256)(inputs)
        x = layers.Dense(self.num_genders, activation=self.gen_act, name="gender_output")(x)
        return x

    def build(self):

        inputShape = (self.height, self.width, self.channel)

        inputs = keras.Input(shape=inputShape)

        x = layers.Conv2D(filters=32, 
                            kernel_size=(3, 3), 
                            kernel_regularizer=l2(l2_rate))(inputs)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.SpatialDropout2D(spatial_dropout_rate_1)(x)
        x = layers.Conv2D(filters=32, 
                            kernel_size=(3, 3), 
                            kernel_regularizer=l2(l2_rate))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.SpatialDropout2D(spatial_dropout_rate_1)(x)
        x = layers.Conv2D(filters=64, 
                            kernel_size=(3,3), 
                            kernel_regularizer=l2(l2_rate))(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)

        x= layers.GlobalAveragePooling2D()(x)

        emotion = self.build_emotion_branch(x)
        gender = self.build_gender_branch(x)

        model = Model(inputs=inputs, outputs=[emotion, gender])

        return model


adnet = AudioNet(height, width).build()
import ipdb; ipdb.set_trace()

losses = {
	"emotion_output": "binary_crossentropy",
	"gender_output": "categorical_crossentropy",
}
lossWeights = {"emotion_output": 1.0, "color_output": 1.0}

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
adnet.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])