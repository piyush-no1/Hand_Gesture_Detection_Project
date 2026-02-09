import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATASET_DIR = "dataset"
MODELS_DIR = "models"

GESTURES = ["up","down","left","right","land","backflip","return_home","spin_180","spin_360","takeoff","forward_up","forward_down", "nod"]

os.makedirs(MODELS_DIR, exist_ok=True)


def compute_extra_features(coords):
    
    fingertips_idx = [4, 8, 12, 16, 20]

    fingertip_dists = [np.linalg.norm(coords[i]) for i in fingertips_idx]

    thumb_tip = coords[4]
    other_tips = [coords[i] for i in [8, 12, 16, 20]]
    thumb_pair_dists = [np.linalg.norm(thumb_tip - tip) for tip in other_tips]

    return np.array(fingertip_dists + thumb_pair_dists, dtype=np.float32)


def preprocess_sample(sample_vec):
    sample_vec = np.asarray(sample_vec, dtype=np.float32)
    if sample_vec.shape[0] != 63:
        raise ValueError(f"Expected 63 features, got {sample_vec.shape[0]}")

    coords = sample_vec.reshape(21, 3)  

    wrist = coords[0].copy()
    coords -= wrist  

    dists = np.linalg.norm(coords, axis=1)  
    max_dist = np.max(dists)
    if max_dist > 0:
        coords /= max_dist

    extra = compute_extra_features(coords)  
   
    coords_flat = coords.flatten()  
    full_features = np.concatenate([coords_flat, extra], axis=0)  
    return full_features


def preprocess_dataset(data):
   
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    processed_rows = []
    for row in data:
        if row.shape[0] != 63:
            continue
        processed_rows.append(preprocess_sample(row))

    if not processed_rows:
        return np.empty((0, 72), dtype=np.float32)

    return np.vstack(processed_rows)


def normalize_features(X):
    
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    return X_norm, mean, std

def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_dataset():
    X_list = []
    y_list = []

    for label_idx, gesture in enumerate(GESTURES):
        csv_path = Path(DATASET_DIR) / f"{gesture}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping this gesture.")
            continue

        raw = np.loadtxt(csv_path, delimiter=",")

        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        data = preprocess_dataset(raw)
        if data.shape[0] == 0:
            print(f"Warning: no valid rows found in {csv_path}, skipping.")
            continue

        print(f"Loaded {data.shape[0]} preprocessed samples for gesture '{gesture}'")

        X_list.append(data)
        y_list.append(np.full((data.shape[0],), label_idx, dtype=np.int64))

    if not X_list:
        raise RuntimeError("No data loaded. Make sure CSV files exist in 'dataset/'.")

    X = np.vstack(X_list).astype("float32")
    y = np.concatenate(y_list)

    print("Total samples:", X.shape[0])
    print("Feature dimension:", X.shape[1])
    return X, y


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X, y = load_dataset()

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

  
    X_norm, mean, std = normalize_features(X)

  
    train_size = int(0.8 * len(X_norm))
    X_train, X_val = X_norm[:train_size], X_norm[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print("Train samples:", X_train.shape[0])
    print("Val samples:", X_val.shape[0])

    model = build_model(input_dim=X_train.shape[1], num_classes=len(GESTURES))

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    model_path = os.path.join(MODELS_DIR, "gesture_ann.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    norm_path = os.path.join(MODELS_DIR, "normalization.npz")
    np.savez(norm_path, mean=mean, std=std)
    print(f"Saved normalization parameters to {norm_path}")

    labels_path = os.path.join(MODELS_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for g in GESTURES:
            f.write(g + "\n")
    print(f"Saved labels to {labels_path}")

if __name__ == "__main__":
    main()