import math
import random
import numpy as np # Import numpy for isnan check if needed, though pandas handles it
import pandas as pd # Import pandas for type hint and isnan check
from torch.utils.data import Sampler, Dataset # Import Dataset for type hint
from collections import Counter, defaultdict

class AtypicalInclusiveBatchSampler(Sampler[list[int]]): # Added type hint
    """
    An improved custom batch sampler that guarantees each batch includes at least one
    atypical sample, if available, while iterating through the dataset exactly once.

    Improvements:
    - Correctly identifies typical samples (where 'atypical' is not 1).
    - More memory efficient by yielding batches directly instead of storing all batches.
    - Adds a 'drop_last' option.

    Expects the dataset to have a 'df' attribute (pandas DataFrame) with an 'atypical' column.
    Values of 1 in the 'atypical' column are considered atypical. Other values (0, False, NaN, None)
    are considered typical.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, debug: bool = False):
        """
        Args:
            dataset: The dataset instance. Must have a 'df' attribute (pd.DataFrame)
                     containing metadata with an 'atypical' column.
            batch_size: The desired batch size.
            shuffle: Whether to shuffle indices at the beginning of each epoch.
            drop_last: If True, drop the last incomplete batch.
            debug: If True, print each generated batch and check atypical inclusion.
        """
        # Input validation
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {batch_size}")
        if not hasattr(dataset, 'df') or not isinstance(dataset.df, pd.DataFrame):
            raise AttributeError("The dataset must have a 'df' attribute of type pandas.DataFrame.")
        if 'atypical' not in dataset.df.columns:
            raise ValueError("The dataset's DataFrame must contain an 'atypical' column.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.debug = debug

        # Identify indices for atypical (== 1) and typical (!= 1) samples
        # Using pandas boolean indexing is generally efficient
        is_atypical = self.dataset.df['atypical'] == 1
        self.atypical_indices = list(self.dataset.df.index[is_atypical])
        # Typical includes False, 0, NaN, None, etc. - anything not explicitly 1
        self.typical_indices = list(self.dataset.df.index[~is_atypical]) # Use negation

        self.num_atypical = len(self.atypical_indices)
        self.num_typical = len(self.typical_indices)
        self.total_samples = self.num_atypical + self.num_typical

        if self.num_atypical == 0:
            print("Warning: No atypical samples found in the dataset.")

        # Calculate the number of batches based on drop_last
        if self.drop_last:
            self.num_batches = self.total_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        # Copy index lists for modification within the iterator
        atypical_pool = self.atypical_indices.copy()
        typical_pool = self.typical_indices.copy()

        if self.shuffle:
            random.shuffle(atypical_pool)
            random.shuffle(typical_pool)

        # Yield batches directly
        batches_yielded = 0
        while atypical_pool or typical_pool:
            # Check if we need to drop the last batch prematurely
            if self.drop_last and batches_yielded >= self.num_batches:
                 break

            batch = []
            # Add one atypical sample if available
            if atypical_pool:
                batch.append(atypical_pool.pop(0))

            # Fill the rest of the batch
            needed = self.batch_size - len(batch)
            can_fill = min(needed, len(typical_pool))

            # Fill with typical samples first
            if can_fill > 0:
                 batch.extend(typical_pool[:can_fill])
                 typical_pool = typical_pool[can_fill:] # More efficient than pop(0) repeatedly

            # If still not full, fill with remaining atypical samples (if any)
            remaining_needed = self.batch_size - len(batch)
            if remaining_needed > 0 and atypical_pool:
                 can_fill_atypical = min(remaining_needed, len(atypical_pool))
                 batch.extend(atypical_pool[:can_fill_atypical])
                 atypical_pool = atypical_pool[can_fill_atypical:]

            # Handle the last batch if drop_last is False and it's incomplete
            if not batch: # Should only happen if both pools were initially empty
                 break
            if not self.drop_last or len(batch) == self.batch_size:
                 if self.debug:
                     print("Yielding Batch indices:", batch)
                     # Retrieve the corresponding atypical flag from the dataset DataFrame.
                     try:
                         batch_labels = self.dataset.df.loc[batch, 'atypical'].tolist()
                         print("Atypical flags in batch:", batch_labels)
                         if 1 not in batch_labels and self.num_atypical > batches_yielded:
                              print("DEBUG WARNING: Batch expected atypical but none found (might be end of epoch).")
                     except KeyError:
                          print("Debug Error: Indices not found in DataFrame.")

                 yield batch
                 batches_yielded += 1
            elif self.drop_last and len(batch) < self.batch_size:
                 # If drop_last is True and the batch is incomplete, stop iteration
                 break


    def __len__(self):
        # Return the pre-calculated number of batches based on drop_last
        return self.num_batches


class PatientAwareAtypicalInclusiveBatchSampler(Sampler[list[int]]):
    """
    A batch sampler that ensures each batch includes at least one atypical sample
    (if available) and attempts to maximize patient diversity within the batch.

    - Prioritizes including one atypical sample.
    - Then, fills the batch with typical samples from unique patients.
    - If needed, allows samples from patients already in the batch (typical first, then atypical).

    Expects the dataset to have a 'df' attribute (pandas DataFrame) with 'atypical'
    and 'Pt ID' columns.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, debug: bool = False):
        """
        Args:
            dataset: The dataset instance. Must have a 'df' attribute (pd.DataFrame)
                     containing 'atypical' and 'Pt ID' columns.
            batch_size: The desired batch size.
            shuffle: Whether to shuffle indices at the beginning of each epoch.
            drop_last: If True, drop the last incomplete batch.
            debug: If True, print detailed batch information including patient IDs.
        """
        # Input validation
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {batch_size}")
        if not hasattr(dataset, 'df') or not isinstance(dataset.df, pd.DataFrame):
            raise AttributeError("The dataset must have a 'df' attribute of type pandas.DataFrame.")
        if 'atypical' not in dataset.df.columns:
            raise ValueError("The dataset's DataFrame must contain an 'atypical' column.")
        if 'Pt ID' not in dataset.df.columns:
            raise ValueError("The dataset's DataFrame must contain a 'Pt ID' column.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.debug = debug

        # Store index-to-patient mapping for quick lookup
        self.index_to_patient_id = self.dataset.df['Pt ID'].to_dict()

        # Identify indices for atypical (== 1) and typical (!= 1) samples
        is_atypical = self.dataset.df['atypical'] == 1
        self.atypical_indices = list(self.dataset.df.index[is_atypical])
        self.typical_indices = list(self.dataset.df.index[~is_atypical])

        self.num_atypical = len(self.atypical_indices)
        self.num_typical = len(self.typical_indices)
        self.total_samples = self.num_atypical + self.num_typical

        if self.num_atypical == 0:
            print("Warning: No atypical samples found in the dataset.")

        # Calculate the number of batches based on drop_last
        if self.drop_last:
            self.num_batches = self.total_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        # Copy index lists for modification within the iterator
        atypical_pool = self.atypical_indices.copy()
        typical_pool = self.typical_indices.copy()

        if self.shuffle:
            random.shuffle(atypical_pool)
            random.shuffle(typical_pool)

        batches_yielded = 0
        while atypical_pool or typical_pool:
            # Check if we need to drop the last batch prematurely
            if self.drop_last and batches_yielded >= self.num_batches:
                 break

            batch = []
            batch_patient_ids = set() # Track patient IDs in the current batch

            # 1. Add one atypical sample if available
            if atypical_pool:
                idx_a = atypical_pool.pop(0)
                batch.append(idx_a)
                patient_id_a = self.index_to_patient_id.get(idx_a)
                if patient_id_a is not None:
                     batch_patient_ids.add(patient_id_a)
                else:
                     print(f"Warning: Patient ID not found for atypical index {idx_a}")

            # 2. Try to fill with typical samples from unique patients
            potential_typical_indices = typical_pool.copy() # Work with a copy for this phase
            indices_added_phase2 = []
            if typical_pool: # Only proceed if there are typicals left
                # Shuffle the potential typicals to avoid bias in selection order
                if self.shuffle:
                    random.shuffle(potential_typical_indices)

                temp_typical_pool = [] # Store indices not used in this phase
                for idx_t in potential_typical_indices:
                    if len(batch) >= self.batch_size:
                        temp_typical_pool.append(idx_t) # Keep for later if batch full
                        continue

                    patient_id_t = self.index_to_patient_id.get(idx_t)
                    if patient_id_t is None:
                         print(f"Warning: Patient ID not found for typical index {idx_t}")
                         # Decide whether to add it anyway or skip
                         # Adding it here for simplicity, but might need review
                         batch.append(idx_t)
                         indices_added_phase2.append(idx_t)
                    elif patient_id_t not in batch_patient_ids:
                         batch.append(idx_t)
                         batch_patient_ids.add(patient_id_t)
                         indices_added_phase2.append(idx_t)
                    else:
                         # Patient already in batch, save index for potential use later
                         temp_typical_pool.append(idx_t)

                # Update the main typical_pool by removing indices added in this phase
                # This is safer than modifying while iterating
                main_typical_set = set(typical_pool)
                added_set = set(indices_added_phase2)
                typical_pool = list(main_typical_set - added_set)
                # Re-shuffle the remaining pool if needed, though order might not matter now
                # if self.shuffle: random.shuffle(typical_pool)


            # 3. Fill remaining slots with skipped typicals (allowing duplicate patients)
            skipped_typical_indices = temp_typical_pool # Indices from patients already in batch
            if self.shuffle: # Shuffle the skipped ones too
                 random.shuffle(skipped_typical_indices)

            fill_needed = self.batch_size - len(batch)
            can_fill_skipped = min(fill_needed, len(skipped_typical_indices))
            if can_fill_skipped > 0:
                 batch.extend(skipped_typical_indices[:can_fill_skipped])
                 # Remove these from the main typical pool as well
                 main_typical_set = set(typical_pool)
                 added_set = set(skipped_typical_indices[:can_fill_skipped])
                 typical_pool = list(main_typical_set - added_set)


            # 4. Fill remaining slots with more atypical samples (allowing duplicate patients)
            fill_needed = self.batch_size - len(batch)
            if fill_needed > 0 and atypical_pool:
                can_fill_atypical = min(fill_needed, len(atypical_pool))
                # Add directly from the start of the shuffled atypical pool
                batch.extend(atypical_pool[:can_fill_atypical])
                atypical_pool = atypical_pool[can_fill_atypical:] # Remove added items


            # Yield the batch if conditions met
            if not batch:
                 break # Stop if no samples could be added
            if not self.drop_last or len(batch) == self.batch_size:
                 if self.debug:
                     print(f"\n--- Yielding Batch {batches_yielded+1} ---")
                     print(" Indices:", batch)
                     try:
                         batch_pids = [self.index_to_patient_id.get(idx, 'N/A') for idx in batch]
                         batch_labels = self.dataset.df.loc[batch, 'atypical'].fillna(-1).tolist() # Use loc, handle NaN
                         print(" Pt IDs: ", batch_pids)
                         print(" Atypical:", batch_labels)
                         # Check if the first sample is atypical if one was expected
                         if self.num_atypical > batches_yielded and batch_labels and batch_labels[0] != 1:
                              print(" DEBUG WARNING: First sample is not atypical, but atypicals should still be available.")
                         # Check patient diversity
                         unique_pids = set(p for p in batch_pids if p != 'N/A')
                         if len(unique_pids) < len(batch):
                              print(f" DEBUG INFO: Batch contains duplicate patients ({len(batch)} samples, {len(unique_pids)} unique patients).")
                     except KeyError as e:
                          print(f"Debug Error: Indices {e} not found in DataFrame during debug print.")
                     except Exception as e:
                          print(f"Debug Error: Unexpected error during debug print: {e}")


                 yield batch
                 batches_yielded += 1
            elif self.drop_last and len(batch) < self.batch_size:
                 # If drop_last is True and the batch is incomplete, stop iteration
                 if self.debug: print(f"Dropping last incomplete batch (size {len(batch)}).")
                 break

    def __len__(self):
        return self.num_batches
        

class PatientAwareAtypicalLabelBalancedSampler(Sampler[list[int]]):
    """
    Custom batch sampler that attempts to:
    1. Guarantee each batch includes at least one 'atypical' sample (if available).
    2. Balance the 'Binary_Label' distribution towards a target ratio.
    3. Maximize patient diversity ('Pt ID') when adding typical samples.

    Expects the dataset to have a 'df' attribute (pandas DataFrame) with
    'atypical', 'Binary_Label', and 'Pt ID' columns.
    """
    def __init__(self, dataset: Dataset, batch_size: int,
                 target_positive_ratio: float = 0.5, # Target ratio for Binary_Label=1
                 shuffle: bool = True, drop_last: bool = False, debug: bool = False):
        """
        Args:
            dataset: The dataset instance. Must have 'df' attribute (pd.DataFrame)
                     containing 'atypical', 'Binary_Label', and 'Pt ID' columns.
            batch_size: The desired batch size.
            target_positive_ratio: Desired proportion of positive (Binary_Label=1) samples.
            shuffle: Whether to shuffle indices at the beginning of each epoch.
            drop_last: If True, drop the last incomplete batch.
            debug: If True, print detailed batch information.
        """
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