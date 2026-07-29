import os
import json
import numpy as np
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'Models')

X_PATH = os.path.join(DATA_DIR, 'X_smart_home.npy')
y_PATH = os.path.join(DATA_DIR, 'y_smart_home.npy')
CLASSES_PATH = os.path.join(DATA_DIR, 'smart_home_classes.json')
MODEL_OUT = os.path.join(MODEL_DIR, 'smart_home_model.h5')

def retrain():
    print("Loading specialized Smart Home data...")
    if not os.path.exists(X_PATH) or not os.path.exists(y_PATH):
        print(f"Error: Could not find data at {X_PATH}. Please collect smart home data first.")
        return

    X = np.load(X_PATH)
    y = np.load(y_PATH)

    with open(CLASSES_PATH, 'r') as f:
        class_map = json.load(f)
    num_classes = len(class_map)

    print(f"Loaded {len(X)} samples across {num_classes} classes.")

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.regularizers import l2
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_OUT):
        backup = MODEL_OUT.replace('.h5', '_backup.h5')
        shutil.copy(MODEL_OUT, backup)
        print(f"Old smart home model backed up → {backup}")

    X_train_raw, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    classes_unique = np.unique(y_train_raw)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_unique,
        y=y_train_raw
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_unique, class_weights)}
    print(f"Computed Class Weights: {class_weight_dict}")

    # Lighter augmentation so GRU does not get overwhelmed
    print("Augmenting training data...")
    X_aug, y_aug = [], []
    for i in range(len(X_train_raw)):
        seq = X_train_raw[i]
        label = y_train_raw[i]

        X_aug.append(seq)
        y_aug.append(label)

        noisy = seq + np.random.normal(0, 0.01, seq.shape)
        X_aug.append(noisy)
        y_aug.append(label)

        shift = np.random.randint(1, 2)
        shifted = np.roll(seq, shift, axis=0)
        shifted[:shift] = seq[0]
        X_aug.append(shifted)
        y_aug.append(label)

    X_train = np.array(X_aug, dtype=np.float32)
    y_train = np.array(y_aug)
    X_test = X_test.astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test_raw, num_classes=num_classes)

    # Better GRU version
    model = Sequential([
        Bidirectional(GRU(
            128,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=l2(0.0005),
            input_shape=(X.shape[1], X.shape[2])
        )),
        BatchNormalization(),
        Dropout(0.2),

        GRU(
            64,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=l2(0.0005)
        ),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_categorical_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5, verbose=1)
    ]

    print("\nStarting Training...")
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    print(f"\nTraining Complete! Best model saved to: {MODEL_OUT}")

if __name__ == "__main__":
    retrain()