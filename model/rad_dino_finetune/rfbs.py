import math
import random
import numpy as np # Import numpy for isnan check if needed, though pandas handles it
import pandas as pd # Import pandas for type hint and isnan check
from torch.utils.data import Sampler, Dataset # Import Dataset for type hint
from collections import Counter, defaultdict

class RepresentationFocusedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: Dataset, batch_size: int,
                 target_positive_ratio: float = 0.5, # Target ratio for Binary_Label=1
                 shuffle: bool = True, drop_last: bool = False, debug: bool = False):

        # --- Input validation ---
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {batch_size}")
        if not (0.0 <= target_positive_ratio <= 1.0):
             raise ValueError(f"target_positive_ratio must be between 0.0 and 1.0, but got {target_positive_ratio}")
        if not hasattr(dataset, 'df') or not isinstance(dataset.df, pd.DataFrame):
            raise AttributeError("The dataset must have a 'df' attribute of type pandas.DataFrame.")
        required_cols = ['atypical', 'Binary_Label', 'Pt ID']
        if not all(col in dataset.df.columns for col in required_cols):
            raise ValueError(f"Dataset DataFrame must contain columns: {required_cols}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.target_positive_ratio = target_positive_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.debug = debug

        # --- Prepare Data & Indices ---
        df = self.dataset.df # Access the DataFrame once
        self.index_to_patient_id = df['Pt ID'].to_dict()
        self.index_to_label = df['Binary_Label'].to_dict()
        self.index_to_atypical = df['atypical'].to_dict()

        # Identify indices for the four groups
        is_atypical = df['atypical'] == 1
        is_positive_label = df['Binary_Label'] == 1

        self.atypical_pos_indices = list(df.index[is_atypical & is_positive_label])
        self.atypical_neg_indices = list(df.index[is_atypical & ~is_positive_label])
        self.typical_pos_indices = list(df.index[~is_atypical & is_positive_label])
        self.typical_neg_indices = list(df.index[~is_atypical & ~is_positive_label])

        # Combine atypical indices for easier initial selection
        self.all_atypical_indices = self.atypical_pos_indices + self.atypical_neg_indices

        self.num_atypical = len(self.all_atypical_indices)
        self.num_typical_pos = len(self.typical_pos_indices)
        self.num_typical_neg = len(self.typical_neg_indices)
        self.total_samples = self.num_atypical + self.num_typical_pos + self.num_typical_neg

        # --- Warnings ---
        if self.num_atypical == 0: print("Warning: No atypical samples found.")
        if self.num_typical_pos == 0: print("Warning: No typical positive samples found.")
        if self.num_typical_neg == 0: print("Warning: No typical negative samples found.")

        # --- Calculate number of batches ---
        if self.drop_last:
            self.num_batches = self.total_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.total_samples / self.batch_size)

    def _get_patient_id(self, index):
        # Helper to safely get patient ID
        return self.index_to_patient_id.get(index, None)

    def _get_label(self, index):
        # Helper to safely get binary label
        return self.index_to_label.get(index, -1) # Return -1 if not found

    def __iter__(self):
        # --- Create shuffled pools for this epoch ---
        atypical_pos_pool = self.atypical_pos_indices.copy()
        atypical_neg_pool = self.atypical_neg_indices.copy()
        typical_pos_pool = self.typical_pos_indices.copy()
        typical_neg_pool = self.typical_neg_indices.copy()
        all_atypical_pool = self.all_atypical_indices.copy()

        if self.shuffle:
            random.shuffle(all_atypical_pool)
            random.shuffle(atypical_pos_pool)
            random.shuffle(atypical_neg_pool)
            random.shuffle(typical_pos_pool)
            random.shuffle(typical_neg_pool)

        # --- Yield batches ---
        batches_yielded = 0
        all_typical_pool = typical_pos_pool + typical_neg_pool # For exhaustion check

        while all_atypical_pool or all_typical_pool:
            if self.drop_last and batches_yielded >= self.num_batches: break

            batch = []
            batch_patient_ids = set()
            num_pos_in_batch = 0
            num_neg_in_batch = 0

            # --- Step 1: Add one atypical sample (if available) ---
            initial_atypical_idx = None
            if all_atypical_pool:
                initial_atypical_idx = all_atypical_pool.pop(0)
                batch.append(initial_atypical_idx)
                pt_id = self._get_patient_id(initial_atypical_idx)
                if pt_id is not None: batch_patient_ids.add(pt_id)
                if self._get_label(initial_atypical_idx) == 1: num_pos_in_batch += 1
                else: num_neg_in_batch += 1
                # Remove from specific sub-pool
                if initial_atypical_idx in atypical_pos_pool: atypical_pos_pool.remove(initial_atypical_idx)
                if initial_atypical_idx in atypical_neg_pool: atypical_neg_pool.remove(initial_atypical_idx)

            # --- Step 2: Try to fill balancing Binary_Label using TYPICAL samples from UNIQUE patients ---
            target_pos_count = round(self.batch_size * self.target_positive_ratio)
            target_neg_count = self.batch_size - target_pos_count

            # Create temporary pools to iterate through without modifying originals yet
            temp_typical_pos = typical_pos_pool.copy()
            temp_typical_neg = typical_neg_pool.copy()
            added_in_step2 = []
            skipped_typicals_due_to_patient = [] # Store indices skipped only due to patient ID conflict

            # Shuffle temp pools if needed
            if self.shuffle:
                random.shuffle(temp_typical_pos)
                random.shuffle(temp_typical_neg)

            pos_idx_iter = 0
            neg_idx_iter = 0
            while len(batch) < self.batch_size and (pos_idx_iter < len(temp_typical_pos) or neg_idx_iter < len(temp_typical_neg)):
                added_this_pass = False
                # Try adding a positive typical from a unique patient
                if num_pos_in_batch < target_pos_count and pos_idx_iter < len(temp_typical_pos):
                    idx_to_check = temp_typical_pos[pos_idx_iter]
                    pt_id = self._get_patient_id(idx_to_check)
                    if pt_id is not None and pt_id not in batch_patient_ids:
                        batch.append(idx_to_check)
                        batch_patient_ids.add(pt_id)
                        num_pos_in_batch += 1
                        added_in_step2.append(idx_to_check)
                        added_this_pass = True
                    elif pt_id is not None: # Patient already in batch
                         skipped_typicals_due_to_patient.append(idx_to_check)
                    # else: pt_id is None, warning? Skip? For now, skip.
                    pos_idx_iter += 1 # Move to next potential positive
                    if len(batch) >= self.batch_size: break # Check if batch full

                # Try adding a negative typical from a unique patient
                if num_neg_in_batch < target_neg_count and neg_idx_iter < len(temp_typical_neg):
                    idx_to_check = temp_typical_neg[neg_idx_iter]
                    pt_id = self._get_patient_id(idx_to_check)
                    if pt_id is not None and pt_id not in batch_patient_ids:
                        batch.append(idx_to_check)
                        batch_patient_ids.add(pt_id)
                        num_neg_in_batch += 1
                        added_in_step2.append(idx_to_check)
                        added_this_pass = True
                    elif pt_id is not None: # Patient already in batch
                         skipped_typicals_due_to_patient.append(idx_to_check)
                    # else: pt_id is None, skip.
                    neg_idx_iter += 1 # Move to next potential negative
                    if len(batch) >= self.batch_size: break # Check if batch full

                # If we couldn't add either positive or negative unique patient this pass, advance both iterators
                if not added_this_pass:
                     if pos_idx_iter < len(temp_typical_pos): pos_idx_iter += 1
                     if neg_idx_iter < len(temp_typical_neg): neg_idx_iter += 1

            # Remove successfully added indices from main pools
            for idx in added_in_step2:
                if idx in typical_pos_pool: typical_pos_pool.remove(idx)
                if idx in typical_neg_pool: typical_neg_pool.remove(idx)

            # --- Step 3: Fill remaining slots with leftover TYPICAL samples (allowing duplicate patients) ---
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0:
                # Combine remaining typicals + those skipped due to patient ID
                combined_typical_leftovers = typical_pos_pool + typical_neg_pool + skipped_typicals_due_to_patient
                if self.shuffle: random.shuffle(combined_typical_leftovers)

                # Prioritize filling based on label balance need
                can_fill = min(remaining_slots, len(combined_typical_leftovers))
                added_in_step3 = []
                temp_pool_step3 = combined_typical_leftovers.copy() # Iterate over copy

                # Try adding positives if needed
                if num_pos_in_batch < target_pos_count:
                    for idx in temp_pool_step3:
                         if len(batch) >= self.batch_size: break
                         if self._get_label(idx) == 1 and idx not in batch: # Check not already added
                             batch.append(idx)
                             added_in_step3.append(idx)
                             num_pos_in_batch += 1
                    # Remove added items from temp pool
                    temp_pool_step3 = [idx for idx in temp_pool_step3 if idx not in added_in_step3]


                # Try adding negatives if needed
                if num_neg_in_batch < target_neg_count and len(batch) < self.batch_size:
                     for idx in temp_pool_step3:
                         if len(batch) >= self.batch_size: break
                         if self._get_label(idx) == 0 and idx not in batch:
                             batch.append(idx)
                             added_in_step3.append(idx)
                             num_neg_in_batch += 1
                     temp_pool_step3 = [idx for idx in temp_pool_step3 if idx not in added_in_step3]

                # Fill remaining slots with anything left from typical leftovers
                if len(batch) < self.batch_size:
                    can_fill_final = min(self.batch_size - len(batch), len(temp_pool_step3))
                    if can_fill_final > 0:
                         final_adds = temp_pool_step3[:can_fill_final]
                         batch.extend(final_adds)
                         added_in_step3.extend(final_adds)

                # Remove all added indices from original pools
                for idx in added_in_step3:
                    if idx in typical_pos_pool: typical_pos_pool.remove(idx)
                    if idx in typical_neg_pool: typical_neg_pool.remove(idx)


            # --- Step 4: Fill any remaining slots with leftover ATYPICAL samples ---
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0:
                 combined_atypical_pool = atypical_pos_pool + atypical_neg_pool
                 if self.shuffle: random.shuffle(combined_atypical_pool)
                 can_fill = min(remaining_slots, len(combined_atypical_pool))
                 if can_fill > 0:
                     added_indices = combined_atypical_pool[:can_fill]
                     batch.extend(added_indices)
                     # Remove from original pools
                     for idx in added_indices:
                         if idx in atypical_pos_pool: atypical_pos_pool.remove(idx)
                         if idx in atypical_neg_pool: atypical_neg_pool.remove(idx)
                         if idx in all_atypical_pool: all_atypical_pool.remove(idx)


            # --- Yield or Break ---
            if not batch: break
            all_typical_pool = typical_pos_pool + typical_neg_pool # Update for loop condition

            if not self.drop_last or len(batch) == self.batch_size:
                if self.debug:
                    print(f"\n--- Yielding Batch {batches_yielded+1} (Size: {len(batch)}) ---")
                    batch_atypical_flags = [self.index_to_atypical.get(idx, -1) for idx in batch]
                    batch_binary_labels = [self.index_to_label.get(idx, -1) for idx in batch]
                    batch_pids_list = [self._get_patient_id(idx) for idx in batch]
                    label_counts = Counter(batch_binary_labels)
                    unique_pids_count = len(set(p for p in batch_pids_list if p is not None))
                    print(f" Indices: {batch}")
                    print(f" Atypical Flags: {batch_atypical_flags}")
                    print(f" Binary Labels: {batch_binary_labels}")
                    print(f" Patient IDs: {batch_pids_list}")
                    print(f" Label Balance: Pos (1): {label_counts[1]}, Neg (0): {label_counts[0]}")
                    print(f" Patient Diversity: {unique_pids_count} unique patients")
                    if 1 not in batch_atypical_flags and self.num_atypical > 0 and batches_yielded < self.num_atypical:
                         print("DEBUG WARNING: Batch expected atypical but none found.")

                yield batch
                batches_yielded += 1
            elif self.drop_last and len(batch) < self.batch_size:
                if self.debug: print(f"Dropping last incomplete batch (size {len(batch)}).")
                break

    def __len__(self):
        return self.num_batches