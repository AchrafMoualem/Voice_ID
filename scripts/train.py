import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

# ==== 1. PARAMETERS ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.h5")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DATASET_PATH = r"C:\Users\hp\Desktop\DATASET"
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40
MAX_PAD_LEN = 100
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4


# ==== 2. DATA AUGMENTATION ====
def augment_audio(y, sr):
    if np.random.rand() > 0.5:
        y = librosa.effects.time_stretch(y, rate=0.8 + 0.4 * np.random.rand())
    if np.random.rand() > 0.5:
        y = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=np.random.choice([-2, -1, 0, 1, 2])
        )
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.005 * np.random.rand(), y.shape)
        y += noise
    if np.random.rand() > 0.5:
        y *= np.random.uniform(0.8, 1.2)
    return y


def augment_mfcc(mfcc):
    if np.random.rand() > 0.5:
        t = np.random.randint(0, mfcc.shape[1])
        mfcc[:, t : t + np.random.randint(5, 15)] = 0
    if np.random.rand() > 0.5:
        f = np.random.randint(0, mfcc.shape[0])
        mfcc[f : f + np.random.randint(2, 6), :] = 0
    return mfcc


# ==== 3. FEATURE EXTRACTION ====
def extract_features(file_path, augment: bool = False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if augment:
            audio = augment_audio(audio, sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        if augment:
            mfcc = augment_mfcc(mfcc)
        mfcc = librosa.util.fix_length(mfcc, size=MAX_PAD_LEN, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==== 4. LOAD DATASET ====
def load_dataset(dataset_path, augment_train: bool = False):
    X, y = [], []
    label_to_index = {}

    for idx, label in enumerate(sorted(os.listdir(dataset_path))):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        label_to_index[label] = idx
        print(f"Processing {label}...")
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(idx)
                if augment_train:
                    features_aug = extract_features(file_path, augment=True)
                    if features_aug is not None:
                        X.append(features_aug)
                        y.append(idx)

    return np.array(X), np.array(y), label_to_index


# ==== 5. MODEL CREATION ====
def create_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==== 6. VISUALIZATION FUNCTIONS ====
def plot_training_history(history_dict, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict["accuracy"], label="Train Acc")
    plt.plot(history_dict["val_accuracy"], label="Val Acc")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_val, y_val, class_names, title="Confusion Matrix"):
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, target_names=class_names))


def plot_roc_auc(model, X_val, y_val, class_names):
    y_score = model.predict(X_val)
    y_val_bin = label_binarize(y_val, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Multi-Class")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_precision_recall(model, X_val, y_val, class_names):
    y_score = model.predict(X_val)
    y_val_bin = label_binarize(y_val, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_val_bin[:, i], y_score[:, i]
        )
        plt.plot(recall, precision, lw=2, label=f"{class_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Multi-Class")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()


# ==== 7. TRAINING WITH CROSS-VALIDATION ====
def train_with_cv(X, y, label_to_index, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_model(X_train.shape[1:], len(label_to_index))
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=4, factor=0.5, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, f"model_fold{fold + 1}.h5"),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        scores.append(val_acc)
        np.save(os.path.join(MODELS_DIR, f"history_fold{fold + 1}.npy"), history.history)

        plot_training_history(history.history, f"Fold {fold + 1}")
        plot_confusion_matrix(
            model,
            X_val,
            y_val,
            list(label_to_index.keys()),
            title=f"Fold {fold + 1}",
        )
        plot_roc_auc(model, X_val, y_val, list(label_to_index.keys()))
        plot_precision_recall(model, X_val, y_val, list(label_to_index.keys()))

        print(f"Fold {fold + 1} - Validation Accuracy: {val_acc:.4f}")

    print(f"\nAverage Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


from sklearn.model_selection import train_test_split


def train_final_model(X, y, label_to_index):
    print(
        "\n🚀 Training final model on 100% of the data (with stratified split)..."
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    model = create_model(X.shape[1:], len(label_to_index))

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUTPUT_PATH,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    gap = history.history["val_accuracy"][-1] - history.history["accuracy"][-1]
    if gap > 0.05:
        print("⚠️ Possible surapprentissage détecté.")
    else:
        print("✅ Pas de surapprentissage significatif.")

    np.save(os.path.join(MODELS_DIR, "history_final1.npy"), history.history)
    plot_training_history(history.history, "Final Model")
    plot_confusion_matrix(
        model, X_val, y_val, list(label_to_index.keys()), title="Final Model"
    )
    plot_roc_auc(model, X_val, y_val, list(label_to_index.keys()))
    plot_precision_recall(model, X_val, y_val, list(label_to_index.keys()))
    print(f"✅ Final model trained and saved as {MODEL_OUTPUT_PATH}")


# ==== 8. MAIN PIPELINE ====
def main():
    X, y, label_to_index = load_dataset(DATASET_PATH, augment_train=True)
    X = X[..., np.newaxis]

    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    np.save(os.path.join(MODELS_DIR, "mean.npy"), mean)
    np.save(os.path.join(MODELS_DIR, "std.npy"), std)
    np.save(os.path.join(MODELS_DIR, "label_mapping.npy"), label_to_index)

    train_with_cv(X, y, label_to_index)
    train_final_model(X, y, label_to_index)


if __name__ == "__main__":
    main()

