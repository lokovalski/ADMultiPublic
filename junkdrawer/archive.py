#######################ARCHIVED CODE FROM MRI BASELINE CLASS####################################
        # def train_and_evaluate_slices(self, test_loader):
    #     if self.train_data is None or self.test_data is None:
    #         return
            
    #     # Train the model
    #     #NOTE ~23% of data has AD so we might want to weight the loss function to balance this
    #     weights = torch.tensor([1.0, 1.0 * (100 - 23) / 23])  # approx [1.0, 3.35]

    #     criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
    #     #criterion = nn.CrossEntropyLoss()

    #     optimizer = torch.optim.Adam(self.model.parameters())

    #     best_acc = float('-inf')
        
    #     print("Training model...")
    #     for epoch in range(8):  
    #         self.model.train()
    #         for batch in tqdm(self.train_data, desc=f"Epoch {epoch+1}"):
    #             mri = batch['mri']  # shape: [batch, channel, slices, H, W] or [batch, channel, slices, H]
    #             labels = batch['label']
    #             num_slices = mri.shape[2]
    #             batch_loss = 0.0

    #             for slice_idx in range(num_slices):
    #                 slice_imgs = mri[:, :, slice_idx, :, :] if mri.dim() == 5 else mri[:, :, slice_idx, :]
    #                 if slice_imgs.dim() == 3:
    #                     slice_imgs = slice_imgs.unsqueeze(1)  # Add channel dim if needed
    #                 outputs = self.model(slice_imgs)
    #                 loss = criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 batch_loss += loss.item()
        
    #     # Evaluate all slices on test set
    #     print("\nEvaluating all slices on test set...")
    #     slice_accuracies = self.evaluate_all_slices(test_loader)
    #     for idx, acc in enumerate(slice_accuracies):
    #         print(f"Slice {idx} accuracy: {acc:.3f}")
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_model = self.model
    #     mean_acc = np.mean(slice_accuracies)
    #     print(f"Mean slice accuracy: {mean_acc:.3f}")


    #     plt.figure(figsize=(12, 6))
    #     plt.bar(range(len(slice_accuracies)), slice_accuracies, label='Per-slice Accuracy')
    #     plt.axhline(y=mean_acc, color='r', linestyle='--', label='Mean Accuracy')
    #     plt.xlabel('Slice Index')
    #     plt.ylabel('Accuracy')
    #     plt.title('MRI Per-slice Accuracy on Test Set')
    #     plt.legend()
    #     plt.savefig('mri_slice_performance.png')
    #     plt.close()
    #     return slice_accuracies, mean_acc

    # def evaluate_slice(self, slice_idx):
    #     if self.test_data is None:
    #         return
            
    #     total_correct = 0
    #     total_samples = 0
        
    #     for batch in self.test_data:
    #         # Fix input dimensions: [batch, channel, height, width]
    #         mri = batch['mri'][:, :, slice_idx:slice_idx+1].squeeze(2)  # Remove the slice dimension
    #         labels = batch['label']
            
    #         with torch.no_grad():
    #             outputs = self.model(mri)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total_correct += (predicted == labels).sum().item()
    #             total_samples += labels.size(0)
        
    #     return total_correct / total_samples if total_samples > 0 else 0

    # def evaluate_plane_average(self, plane_slices):
    #     """Evaluate model performance on average of slices from a specific plane"""
    #     if self.test_data is None:
    #         return
            
    #     total_correct = 0
    #     total_samples = 0
        
    #     for batch in self.test_data:
    #         # Average the specified slices and fix dimensions
    #         mri = torch.mean(batch['mri'][:, :, plane_slices], dim=2)  # Average over slices
    #         labels = batch['label']
            
    #         with torch.no_grad():
    #             outputs = self.model(mri)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total_correct += (predicted == labels).sum().item()
    #             total_samples += labels.size(0)
        
    #     return total_correct / total_samples if total_samples > 0 else 0
    
    # def evaluate_all_slices(self, loader):
    #     """Evaluate model performance for each slice index across all samples in the loader."""

    #     num_slices = None
    #     for batch in loader:
    #         mri = batch['mri']
    #         num_slices = mri.shape[2]
    #         break

    #     correct = np.zeros(num_slices)
    #     total = np.zeros(num_slices)

    #     for batch in loader:
    #         mri = batch['mri'].to(self.device)
    #         labels = batch['label'].to(self.device).squeeze()

    #         for slice_idx in range(num_slices):
    #             # Extract the slice for all samples in the batch
    #             slice_imgs = mri[:, :, slice_idx, :, :] if mri.dim() == 5 else mri[:, :, slice_idx, :]
                
    #             if slice_imgs.dim() == 3:
    #                 slice_imgs = slice_imgs.unsqueeze(1)  # add back channel dim if needed

    #             with torch.no_grad():
    #                 outputs = self.model(slice_imgs)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 correct[slice_idx] += (predicted == labels).sum().item()
    #                 total[slice_idx] += labels.size(0)

    #     return correct/total


#######################ARCHIVED CODE FROM NOTEBOOK####################################
# print("\nTraining MRI baseline...")
# mri_model = MRIBaseline()
# mri_model.load_data(train_loader, val_loader)
# # Debug: check if slices are unique in the test set
# # mri_model.debug_slices(test_loader)
# slice_accuracies, mean_acc = mri_model.train_and_evaluate_slices(test_loader)

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(slice_accuracies)), slice_accuracies, label='Individual Slices')
# plt.axhline(y=np.mean(slice_accuracies), color='r', linestyle='--', label='Average Slice Performance')
# plt.xlabel('Slice Index')
# plt.ylabel('Accuracy')
# plt.title('MRI Slice Performance')
# plt.legend()
# plt.savefig('mri_slice_performance.png')
# plt.close()