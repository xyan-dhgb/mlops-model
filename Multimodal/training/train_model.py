import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def train_model(model, X_tab_train, X_img_train, y_train, X_tab_val, X_img_val, y_val, epochs=20, batch_size=32):
    """
    Train the multimodal model
    """
    # Data augmentation for images
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        [X_tab_train, X_img_train],
        y_train,
        validation_data=([X_tab_val, X_img_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    return model, history