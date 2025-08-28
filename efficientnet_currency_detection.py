import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse

# --- 1. Configuration and Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths (assuming the same structure as the notebook)
TRAIN_DIR = "dataset/training"
VALIDATION_DIR = "dataset/validation"
MODEL_SAVE_PATH = "Final_model_efficientnet.pth" # Use .pth for PyTorch

# --- Pre-run check for dataset directories ---
if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALIDATION_DIR):
    print("Error: Training or validation directory not found.")
    print(f"Please make sure the '{TRAIN_DIR}' and '{VALIDATION_DIR}' directories exist.")
    exit()

# Model and Training parameters
IMG_HEIGHT = 224 # EfficientNet-B0 preferred size
IMG_WIDTH = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20 # Increased epochs for a more gradual schedule
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CLASS_LIST = ['Fake', 'Real']
NUM_CLASSES = len(CLASS_LIST)

# --- 2. Data Loading and Transformations ---

# EfficientNet models require specific normalization. We get the official transforms.
weights = models.EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms()

# For training, we add the augmentations from the original notebook
train_transforms = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    preprocess, # This includes resizing to 224x224 and normalization
])

# For validation, we only need the basic preprocessing
validation_transforms = preprocess

# Create datasets using ImageFolder
train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
validation_dataset = ImageFolder(root=VALIDATION_DIR, transform=validation_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Found {len(train_dataset)} images in training folder, belonging to {len(train_dataset.classes)} classes.")
print(f"Found {len(validation_dataset)} images in validation folder, belonging to {len(validation_dataset.classes)} classes.")
print(f"Classes: {train_dataset.class_to_idx}")


# --- 3. Model Definition ---

def create_finetune_model(num_classes, weights_enum=models.EfficientNet_B0_Weights.DEFAULT):
    """Creates, customizes, and returns the EfficientNet model for fine-tuning."""
    # Load a pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=weights_enum)

    # Freeze all the parameters in the feature extraction part of the model if using pre-trained weights
    if weights_enum is not None:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier with a new one that matches the Keras architecture
    # Original Keras: Flatten, Dense(1024), Dropout(0.5), Dense(1024), Dropout(0.5), Dense(2)
    in_features = model.classifier[1].in_features
    FC_Layers = [1024, 1024]
    dropout = 0.5

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, FC_Layers[0]), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(FC_Layers[0], FC_Layers[1]), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(FC_Layers[1], num_classes)
    )
    return model

# Create the model for training
model = create_finetune_model(NUM_CLASSES, weights_enum=weights)
model = model.to(DEVICE)
print("Model architecture updated for fine-tuning.")

# --- 4. Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# Note: We only pass the parameters of the new classifier to the optimizer.

# Add a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5) # Less aggressive decay

# --- 5. Training Loop ---
def train_model():
    best_val_accuracy = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}\n' + '-'*10)
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(validation_dataset)
        val_epoch_acc = val_corrects.double() / len(validation_dataset)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        # Step the scheduler
        scheduler.step()

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # Save the model if it has the best validation accuracy so far
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'New best model saved to {MODEL_SAVE_PATH} (Val Acc: {val_epoch_acc:.4f})')

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_accuracy:4f}')
    
    # Plot training history
    plot_training_history(history)

    # Automatically evaluate the best model after training
    evaluate_model()

# --- 6. Plotting and Evaluation ---
def plot_training_history(history):
    """Plots the training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')

    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper left')

    plt.savefig("training_history_plot.png")
    plt.show()

def evaluate_model():
    """Loads the best model and evaluates it on the validation set."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file not found at {MODEL_SAVE_PATH}. Please train the model first.")
        return

    print("\n--- Evaluating model on validation set ---")
    # Load the best model for evaluation
    model = create_finetune_model(NUM_CLASSES, weights_enum=None)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=CLASS_LIST, digits=4)
    print(report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LIST)
    disp.plot()
    plt.show()


def test_single_image(image_path):
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file not found at {MODEL_SAVE_PATH}. Please train the model first.")
        return

    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}. Please check the path.")
        return

    # Load the model with the best weights
    # We need to define the model architecture again before loading the state dict
    test_model = create_finetune_model(NUM_CLASSES, weights_enum=None) # No pre-trained weights needed
    test_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_model = test_model.to(DEVICE)
    test_model.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = validation_transforms(img).unsqueeze(0).to(DEVICE)

    # Display the image
    plt.imshow(img)
    plt.show()

    # Predict
    with torch.no_grad():
        output = test_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred_idx = torch.max(output, 1)
        predicted_class = CLASS_LIST[pred_idx.item()]

    print(f"Prediction: {predicted_class}")
    print(f"Probabilities -> Fake: {probabilities[0]:.4f}, Real: {probabilities[1]:.4f}")


if __name__ == '__main__':
    # --- 8. Command-Line Interface ---
    parser = argparse.ArgumentParser(description='Fake Currency Detection with EfficientNet.')
    parser.add_argument('--train', action='store_true', help='Flag to train the model.')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model on the validation set.')
    parser.add_argument('--test', type=str, metavar='PATH', help='Flag to test a single image. Provide the image path.')

    args = parser.parse_args()

    if args.train:
        print("--- Starting Model Training ---")
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.test:
        print(f"--- Testing single image: {args.test} ---")
        test_single_image(image_path=args.test)
    else:
        print("No action specified. Please use --train, --evaluate, or --test <path_to_image>.")
