import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def build_multimodal_model(tabular_shape, image_shape, num_classes=3):
    """
    Build a multimodal model that combines tabular data and images
    """
    # Tabular data branch
    tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
    x_tab = Dense(64, activation='relu')(tabular_input)
    x_tab = Dropout(0.3)(x_tab)
    x_tab = Dense(32, activation='relu')(x_tab)
    x_tab = Model(inputs=tabular_input, outputs=x_tab)

    # Image branch using pretrained model
    image_input = Input(shape=image_shape, name='image_input')

    # Convert greyscale to RGB by repeating the channel
    x_img = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(image_input)

    # Use pretrained ResNet with weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    x_img = base_model(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(128, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)
    x_img = Model(inputs=image_input, outputs=x_img)

    # Combine the branches
    combined = concatenate([x_tab.output, x_img.output])

    # Final layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=[x_tab.input, x_img.input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model