import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

DATA_DIR = 'Data'
MODEL_DIR = 'Models'

def train_model():
    X_path = os.path.join(DATA_DIR, 'X_combined.npy')
    y_path = os.path.join(DATA_DIR, 'y_combined.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Data files not found. Run data_merger.py first.")
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Loaded data: X={X.shape}, y={y.shape}")
    
    # Advanced: Stratified Split to ensure fair testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = Sequential()
    # 1. Bidirectional LSTM: Context from both past and future
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2)) # Regularization
    
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    # 2. L2 Regularization to preventing neuron dominance
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    # Gradient Clipping to prevent exploding gradients
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    
    save_path = os.path.join(MODEL_DIR, 'final_model.h5')
    cp_callback = ModelCheckpoint(save_path, monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Early Stopping: Stop if no improvement for 10 epochs
    es_callback = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, restore_best_weights=True)
    
    # Reduce LR: Lower learning rate if stuck for 5 epochs
    lr_callback = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, verbose=1, min_lr=0.00001)
    
    model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback, cp_callback, es_callback, lr_callback], validation_data=(X_test, y_test))
    
    model.summary()
    print(f"Best model saved to {save_path}")

if __name__ == "__main__":
    train_model()
