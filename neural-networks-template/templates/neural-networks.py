import tensorflow as tf

from sklearn.model_selection import train_test_split

X = df.drop(['user-definedlabeln'], axis=1)
y = df['user-definedlabeln']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = tf.keras.models.Sequential([
        keras.layers.Dense(units=32, activation="sigmoid"),
        keras.layers.Dense(units=16, activation="sigmoid"),
        keras.layers.Dense(units=1, activation="ReLU")
    ])

from tensorflow import RMSprop
from keras.optimizers import RMSprop
model.compile(optimizer= RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])


from keras.optimizers import RMSprop
model.compile(optimizer= RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])


model.fit(X_train, y_train, epochs = 15, batch_size = 10)

model.evaluate(X_test, y_test)

model.predict(predict_data) 