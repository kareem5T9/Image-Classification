import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_DIR = r"dataset/10_big_cats"  # Raw string for Windows paths
TARGET_SIZE = (64, 64)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (3, 3),
    'channel_axis': None
}

def load_data(data_dir, target_size=TARGET_SIZE):
    """Load and preprocess images with validation checks"""
    images = []
    labels = []
    
    # Validate data directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    class_names = sorted(os.listdir(data_dir))
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"Found {len(class_names)} classes: {class_names}")

    # Load images
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Skipping non-directory: {class_dir}")
            continue
            
        image_files = os.listdir(class_dir)
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
            
        print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = imread(img_path)
                gray_img = rgb2gray(img)
                resized_img = resize(gray_img, target_size, anti_aliasing=True)
                images.append(resized_img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue

    if len(images) == 0:
        raise ValueError("No images loaded! Check dataset structure and files.")
    
    print(f"\nSuccessfully loaded {len(images)} images from {len(class_names)} classes")
    return np.array(images), np.array(labels), class_names

def visualize_samples(images, labels, class_names, n_samples=5):
    """Display random samples from each class"""
    plt.figure(figsize=(15, 10))
    unique_labels = np.unique(labels)
    
    for label_idx, class_label in enumerate(unique_labels):
        class_images = images[labels == class_label]
        sample_indices = np.random.choice(len(class_images), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt_idx = label_idx * n_samples + i + 1
            plt.subplot(len(unique_labels), n_samples, plt_idx)
            plt.imshow(class_images[idx], cmap='gray')
            plt.title(class_names[class_label])
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_hog_features(image, hog_params=HOG_PARAMS):
    """Visualize HOG features for a sample image"""
    _, hog_image = hog(image, visualize=True, **hog_params)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_hog_features(images, hog_params=HOG_PARAMS):
    """Extract HOG features from images"""
    features = []
    for img in images:
        fd = hog(img, **hog_params)
        features.append(fd)
    return np.array(features)

def preprocess_data(X_train, X_test):
    """Apply standardization and PCA"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_test_pca

def train_model(X_train, y_train):
    """Train SVM classifier"""
    svm = SVC(C=10, kernel='rbf', gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Load and validate data
    images, labels, class_names = load_data(DATA_DIR)
    
    # Visualize samples
    visualize_samples(images, labels, class_names)
    
    # Show HOG features for random image
    random_idx = np.random.randint(len(images))
    visualize_hog_features(images[random_idx])
    
    # Feature extraction
    hog_features = extract_hog_features(images)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        hog_features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Dimensionality reduction
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test)
    
    # Train model
    model = train_model(X_train_pca, y_train)
    
    # Evaluate performance
    evaluate_model(model, X_test_pca, y_test, class_names)