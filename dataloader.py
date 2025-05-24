import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class AlzheimerDataset(Dataset):
    # Note: target size is chosen to accomodate pretrained VGG16 model
    def __init__(self, data_root_directory, target_size=(224, 224)):
        self.data_root_directory = data_root_directory
        self.target_size = target_size
        self.mri_paths = []
        self.tabular_data = []
        self.labels = []
        self.multiclass_labels = []
        
        # Find all subject directories
        for subject_dir in sorted(glob.glob(os.path.join(data_root_directory, "disc*/*_MR1"))):
            mri_path = os.path.join(subject_dir, 'PROCESSED/MPRAGE/T88_111',
                                  f"{os.path.basename(subject_dir)}_mpr_n4_anon_111_t88_gfc.img")
            txt_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}.txt")
            if os.path.exists(mri_path) and os.path.exists(txt_path):
                # Parse features
                data = {}
                with open(txt_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            data[key.strip().upper()] = value.strip()
                features = {
                    'age': self.str_2_float(data.get('AGE')),
                    'sex': 1 if data.get('M/F', '').lower().startswith('f') else 0, 
                    'education': self.str_2_float(data.get('EDUC')),
                    'mmse': self.str_2_float(data.get('MMSE')),
                    'hand': self.str_2_float(data.get('HAND')),
                    'etiv': self.str_2_float(data.get('ETIV')),
                    'asf': self.str_2_float(data.get('ASF')),
                    'nwbv': self.str_2_float(data.get('NWBV'))
                }
                # Check for unrecoverable missing data
                feature_vals = np.array(list(features.values()), dtype=np.float32)
                if np.isnan(feature_vals).all():
                    continue  # skip this subject
                # Save CDR as multiclass label for use later potentially
                cdr_val = self.str_2_float(data.get('CDR'))
                multiclass_label = 0 if np.isnan(cdr_val) else cdr_val
                label = 1 if self.str_2_float(data.get('CDR')) > 0 else 0
                self.mri_paths.append(mri_path)
                self.tabular_data.append(features)
                self.labels.append(label)
                self.multiclass_labels.append(multiclass_label)
    
    def parse_text_file(self, txt_path):
        """Extract tabular data and label from text file"""
        data = {}
        with open(txt_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    data[key.strip().upper()] = value.strip()

        features = {
            'age': self.str_2_float(data.get('AGE')),
            'sex': 1 if data.get('M/F', '').lower().startswith('f') else 0, 
            'education': self.str_2_float(data.get('EDUC')),
            'mmse': self.str_2_float(data.get('MMSE')),
            'hand': self.str_2_float(data.get('HAND')),
            'etiv': self.str_2_float(data.get('ETIV')),
            'asf': self.str_2_float(data.get('ASF')),
            'nwbv': self.str_2_float(data.get('NWBV'))
        }
        self.tabular_data.append(features)

        # Save CDR as multiclass label for use later potentially
        cdr_val = self.str_2_float(data.get('CDR'))
        self.multiclass_labels.append(0 if np.isnan(cdr_val) else cdr_val)

        # Convert CDR into binary label using cutoff provided in docs. 1 = demented 
        self.labels.append(1 if self.str_2_float(data.get('CDR')) > 0 else 0)
    

    # This is just to deal with NaNs nicely. Ignore
    @staticmethod
    def str_2_float(value, default=np.nan):
        if value is None:
            return default
        value = value.strip()
        if value == '' or value.lower() == 'nan':
            return default
        try:
            return float(value)
        except Exception:
            return default
    
    def process_slice(self, slice_data):
        """Process a single MRI slice"""
        
        #If slice_data is 3D, squeeze to 2D
        if slice_data.ndim == 3:
            slice_data = np.squeeze(slice_data)
            # If still 3D (e.g., shape (x, y, 1)), take the first channel
            if slice_data.shape[-1] == 1:
                slice_data = slice_data[..., 0]

        if slice_data.ndim != 2:
            raise ValueError(f"Slice data must be 2D, got shape {slice_data.shape}")
        
        #Normalize to [0, 255] just to print image nicely
        if np.max(slice_data) > np.min(slice_data):
            slice_data = ((slice_data - np.min(slice_data)) /
                         (np.max(slice_data) - np.min(slice_data)) * 255)
            
        slice_data = slice_data.astype('uint8')
        slice_img = Image.fromarray(slice_data, mode='L')
        slice_img = slice_img.resize(self.target_size, Image.BILINEAR)
        return np.array(slice_img, dtype=np.float32)
    
    def load_mri(self, mri_path):
        """Load and process MRI data
        
        1. Loads MRI file using nibabel and extracts data as a numpyarray
        2. Extracts 9 slices (3 from each plane: axial, sagittal, and coronal)
        3. Normalizes each slice 
        4. Stacks slices into a single numpy array of shape (9,target_shape[0],target_shape[1]) 
        5. Add channel dimension for CNN input
        """

        #recall that MRIs are in Analyze 7.5 format, which nibabel can load
        # get_fdata() extract data as numpy array
        data = nib.load(mri_path).get_fdata()

        if data.ndim == 4:
            data = data[..., 0]
        positions = [0.4, 0.5, 0.6]
        slices = []
        x, y, z = data.shape[:3]

        # Axial (top-down)
        for pos in positions:
            slices.append(self.process_slice(data[:, :, int(z * pos)]))

        # Sagittal (side)
        for pos in positions:
            slices.append(self.process_slice(data[int(x * pos), :, :]))

        # Coronal (front)
        for pos in positions:
            slices.append(self.process_slice(data[:, int(y * pos), :]))

        #Normalize to [0,1] now
        slices = np.stack(slices) / 255.0
        return np.expand_dims(slices, 0)  
    
    def __len__(self):
        return len(self.mri_paths)
    
    def __getitem__(self, idx):
        ''' 
        Gives all the info for a single subject
        Returns dictionary with keys: 'mri', 'tabular', 'label', 'multiclass_label'
        '''
        mri = torch.FloatTensor(self.load_mri(self.mri_paths[idx]))
        tabular = np.array([
            self.tabular_data[idx][key] for key in [
                'age', 'sex', 'education', 'mmse', 'hand', 'etiv', 'asf', 'nwbv']
        ], dtype=np.float32)

        if not hasattr(self, '_feature_means'):
            all_tabular = np.array([[row[key] for key in ['age', 'sex', 'education', 'mmse', 'hand', 'etiv', 'asf', 'nwbv']] for row in self.tabular_data], dtype=np.float32)
            if all_tabular.size == 0 or len(all_tabular) == 0:
                # No data: set to zeros of the right shape
                self._feature_means = np.zeros(8, dtype=np.float32) 
            else:
                # Only compute mean if we have data
                self._feature_means = np.nanmean(all_tabular, axis=0)
                # Replace any remaining NaN means with 0
                self._feature_means = np.nan_to_num(self._feature_means, nan=0.0)
                
        inds = np.where(np.isnan(tabular))[0]
        if len(inds) > 0:
            tabular[inds] = self._feature_means[inds]
        # If any feature is still nan (e.g., all values were nan for that feature), just return as is (should be rare)
        return {
            'mri': mri,
            'tabular': torch.FloatTensor(tabular),
            'label': torch.LongTensor([self.labels[idx]]),
            'multiclass_label': torch.LongTensor([self.multiclass_labels[idx]])
        }

def get_data_loaders(data_root_directory, batch_size=32, num_workers=4):
    """
    Create train, validation, and test data loaders
    
    data_root_directory (str): Root directory containing the discs
    batch_size: Batch size for data loaders
    num_workers: Number of workers for data loading
    
    train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing
    """
    dataset = AlzheimerDataset(data_root_directory)
    
    # Choosing to do 70/15/15 split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    # Load data for each split using torch's DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader