"""Aircraft damage classification and image captioning pipeline.

It covers two tasks:
1. Binary image classification of aircraft damage (dent vs crack) using VGG16.
2. Image caption/summary generation using Salesforce BLIP.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import BlipForConditionalGeneration, BlipProcessor


DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
)
DEFAULT_TAR_PATH = Path("aircraft_damage_dataset_v1.tar")
DEFAULT_EXTRACT_DIR = Path("aircraft_damage_dataset_v1")


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 5
    img_height: int = 224
    img_width: int = 224
    learning_rate: float = 1e-4
    seed: int = 42

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.img_height, self.img_width, 3)


def set_seed(seed: int) -> None:
    """Set all major random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_and_extract_dataset(
    url: str = DATASET_URL,
    tar_path: Path = DEFAULT_TAR_PATH,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
    force_redownload: bool = False,
) -> Path:
    """Download and extract the aircraft damage dataset."""
    if force_redownload or not tar_path.exists():
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded: {tar_path}")

    if extract_dir.exists():
        print(f"Removing existing extracted directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    print(f"Extracting dataset to: {extract_dir}")
    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall()

    print("Dataset ready.")
    return extract_dir


def create_data_generators(
    dataset_dir: Path,
    config: Config,
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, ...]:
    """Create training, validation, and test generators."""
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_args = dict(
        target_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        class_mode="binary",
        seed=config.seed,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        shuffle=True,
        **common_args,
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        shuffle=False,
        **common_args,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        shuffle=False,
        **common_args,
    )

    return train_generator, valid_generator, test_generator


def build_classifier(config: Config) -> tf.keras.Model:
    """Build a VGG16-based binary classifier."""
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.input_shape,
    )

    output = base_model.layers[-1].output
    output = Flatten()(output)
    feature_extractor = Model(base_model.input, output)

    for layer in feature_extractor.layers:
        layer.trainable = False

    classifier = Sequential(
        [
            feature_extractor,
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    classifier.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return classifier


def train_classifier(
    model: tf.keras.Model,
    train_generator,
    valid_generator,
    epochs: int,
) -> tf.keras.callbacks.History:
    """Train the classifier."""
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
    )
    return history


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation loss/accuracy curves."""
    metrics = history.history

    plots = [
        ("loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
        ("accuracy", "Training Accuracy"),
        ("val_accuracy", "Validation Accuracy"),
    ]

    for metric_name, title in plots:
        if metric_name in metrics:
            plt.figure(figsize=(6, 4))
            plt.plot(metrics[metric_name])
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.show()


def evaluate_classifier(model: tf.keras.Model, test_generator) -> Dict[str, float]:
    """Evaluate the classifier on the test set."""
    test_steps = max(1, test_generator.samples // test_generator.batch_size)
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
    results = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    return results


def plot_image_with_title(
    image_array: np.ndarray,
    true_label: int,
    predicted_label: int,
    class_names: Dict[int, str],
) -> None:
    """Plot a single image with its true and predicted labels."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array)
    plt.title(
        f"True: {class_names[int(true_label)]}\n"
        f"Pred: {class_names[int(predicted_label)]}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def preview_prediction(test_generator, model: tf.keras.Model, index_to_plot: int = 0) -> None:
    """Show one prediction from the test generator."""
    test_generator.reset()
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    class_names = {v: k for k, v in test_generator.class_indices.items()}
    plot_image_with_title(
        image_array=test_images[index_to_plot],
        true_label=int(test_labels[index_to_plot]),
        predicted_label=int(predicted_classes[index_to_plot]),
        class_names=class_names,
    )


class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    """Custom TensorFlow layer that wraps BLIP caption generation."""

    def __init__(self, processor: BlipProcessor, model: BlipForConditionalGeneration, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        return tf.py_function(self.process_image, [image_path, task], Tout=tf.string)

    def process_image(self, image_path, task):
        try:
            image_path_str = image_path.numpy().decode("utf-8")
            task_name = task.numpy().decode("utf-8")

            image = Image.open(image_path_str).convert("RGB")
            prompt = (
                "This is a picture of"
                if task_name == "caption"
                else "This is a detailed photo showing"
            )

            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            output = self.model.generate(**inputs)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Error while processing image: {exc}")
            return "Error processing image"


class BlipService:
    """Helper service for image captioning and summarization."""

    def __init__(self) -> None:
        print("Loading BLIP processor and model...")
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.layer = BlipCaptionSummaryLayer(self.processor, self.model)

    def generate_text(self, image_path: str | Path, task: str = "caption") -> str:
        result = self.layer(tf.constant(str(image_path)), tf.constant(task))
        return result.numpy().decode("utf-8")


def display_image(image_path: str | Path) -> None:
    """Display an image using matplotlib."""
    img = plt.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_classification_pipeline(config: Config, dataset_dir: Path) -> None:
    """Run the full classification workflow."""
    train_generator, valid_generator, test_generator = create_data_generators(dataset_dir, config)
    classifier = build_classifier(config)
    history = train_classifier(classifier, train_generator, valid_generator, config.epochs)
    plot_history(history)
    evaluate_classifier(classifier, test_generator)
    preview_prediction(test_generator, classifier, index_to_plot=1)


def run_caption_pipeline(image_path: str | Path) -> None:
    """Run BLIP caption and summary generation for a single image."""
    blip_service = BlipService()
    display_image(image_path)

    caption = blip_service.generate_text(image_path, task="caption")
    summary = blip_service.generate_text(image_path, task="summary")

    print(f"Caption: {caption}")
    print(f"Summary: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aircraft damage classification and captioning pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "caption", "all"],
        default="all",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for the classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data generators.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=(
            "aircraft_damage_dataset_v1/test/dent/"
            "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
        ),
        help="Image path for BLIP caption generation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download/extraction if it is already available locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(batch_size=args.batch_size, epochs=args.epochs)
    set_seed(config.seed)

    dataset_dir = DEFAULT_EXTRACT_DIR
    if not args.skip_download or not dataset_dir.exists():
        dataset_dir = download_and_extract_dataset()

    if args.mode in {"classify", "all"}:
        run_classification_pipeline(config, dataset_dir)

    if args.mode in {"caption", "all"}:
        run_caption_pipeline(args.image_path)


if __name__ == "__main__":
    main()
