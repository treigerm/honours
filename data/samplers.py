import torch
import numpy as np
import random

CASE_INDEX = 2 # Column index of case ID in metadata table.

def safe_pop(l, k):
    """Pops min(len(l), k) elements from l.
    """
    samples = []
    while len(l) > 0 and len(samples) < k:
        samples.append(l.pop(random.randrange(len(l))))
    return samples

def make_batch_sampler(data_source, 
                       slides_frame, 
                       cases_per_batch, 
                       patches_per_case, 
                       drop_last=False):
    case_sampler = CaseSampler(
        data_source, slides_frame, cases_per_batch, patches_per_case)
    return torch.utils.data.BatchSampler(
        case_sampler, case_sampler.batch_size, drop_last)

class CaseSampler(torch.utils.data.Sampler):
    """Sampler for the PatchWiseDataset. 
    
    It creates an iterator that can be used with a torch.utils.data.BatchSampler
    object with a specific batch size. The resulting batches will each have 
    cases_per_batch cases and for each cases patches_per_case patches."""

    def __init__(self, 
                 data_source, 
                 slides_frame, 
                 cases_per_batch, 
                 patches_per_case,
                 num_samples=None):
        self.data_source = data_source
        self.slides_frame = slides_frame
        self.cases_per_batch = cases_per_batch
        self.patches_per_case = patches_per_case
        self.num_samples = num_samples
        self.batch_size = cases_per_batch * patches_per_case

    
    def __iter__(self):
        # Create a dictionary {case_id: [index]} so the indexes are 
        # grouped by case ID.
        cases = dict()
        for i in range(len(self.data_source)):
            idx, _, _, _, _, _ = np.unravel_index(i, self.data_source.shape)
            case_id = self.slides_frame.iloc[idx, CASE_INDEX]
            if case_id in cases:
                cases[case_id].append(i)
            else:
                cases[case_id] = [i]
        
        ix_samples = []
        while len(cases) > 0:
            num_cases = min(self.cases_per_batch, len(cases))
            # TODO: Sample based on case
            batch_cases = random.sample(cases.keys(), num_cases)
            for c in batch_cases:
                ix_samples += safe_pop(cases[c], self.patches_per_case)
                if len(cases[c]) == 0:
                    cases.pop(c, None)
                
        if self.num_samples is not None:
            return iter(ix_samples[:self.num_samples])

        return iter(ix_samples)

    def __len__(self):
        return len(self.data_source)