from keras.layers import Activation, Dense, Dropout, Flatten

from keras.models import Sequential
model = Sequential()
model.add(Dense(408, input_shape=(16, 16, 1000), init='uniform'))
model.add(Activation('relu'))

model.add(
    Dense(
        204, init='uniform', W_regularizer=l2(0.003), b_regularizer=l2(0.003)
    )
)
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(
    Dense(51, init='uniform', W_regularizer=l2(0.003), b_regularizer=l2(0.003))
)
model.add(Activation('relu'))

model.add(
    Dense(
        classes,
        init='uniform',
        W_regularizer=l2(0.003),
        b_regularizer=l2(0.003)
    )
)
model.add(Activation('softmax'))
