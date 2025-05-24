import matplotlib.pyplot as plt
import numpy as np
from dataloader import AlzheimerDataset

def test_first_patient():
    # Load first patient's data
    dataset = AlzheimerDataset('data')
    sample = dataset[0]
    
    # Print patient information
    print("\n=== Patient Information ===")
    features = ['Age', 'Sex', 'Education', 'MMSE', 'CDR']
    values = sample['tabular'].numpy()
    for feature, value in zip(features, values):
        if feature == 'Sex':
            print(f"{feature}: {'Female' if value == 1 else 'Male'}")
        else:
            print(f"{feature}: {value:.1f}")
    print(f"Label: {'Dementia' if sample['label'].item() == 1 else 'No Dementia'}")
    
    # Visualize MRI slices
    mri_data = sample['mri'].numpy()
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    planes = ['Axial (Top-Down)', 'Sagittal (Side)', 'Coronal (Front)']
    positions = ['40%', '50%', '60%']
    
    for i, plane in enumerate(planes):
        for j, pos in enumerate(positions):
            ax = axes[i, j]
            ax.imshow(mri_data[0, i*3 + j], cmap='gray')
            title = f'{plane}\n{pos}' if j == 0 else pos
            ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_first_patient() 