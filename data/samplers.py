import torch
import numpy as np
import random

CASE_INDEX = 2 # Column index of case ID in metadata table.

def safe_pop(l, k):
    """Pops min(len(l), k) elements from l.
    """
    num_samples = 0
    while len(l) > 0 and num_samples < k:
        num_samples += 1
        yield l.pop(random.randrange(len(l)))

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
        
        #ix_samples = []
        num_samples = 0
        max_samples = float("inf") if self.num_samples is None else self.num_samples
        while len(cases) > 0:
            num_cases = min(self.cases_per_batch, len(cases))
            # TODO: Sample based on case
            batch_cases = random.sample(cases.keys(), num_cases)
            for c in batch_cases:
                #ix_samples += safe_pop(cases[c], self.patches_per_case)
                for sample in safe_pop(cases[c], self.patches_per_case):
                    if num_samples >= max_samples:
                        return
                    num_samples += 1
                    yield sample

                if len(cases[c]) == 0:
                    # Remove case if it does not have any tiles left.
                    cases.pop(c, None)
                
    def __len__(self):
        return len(self.data_source)

class CaseUniqueSampler(torch.utils.data.Sampler):
    """Sample each case exactly once"""

    def __init__(self, 
                 data_source, 
                 slides_frame, 
                 patches_per_case,
                 num_samples=None):
        self.data_source = data_source
        self.slides_frame = slides_frame
        self.patches_per_case = patches_per_case
        self.num_samples = num_samples
        self.batch_size = patches_per_case

        # Create a dictionary {case_id: [index]} so the indexes are 
        # grouped by case ID.
        self.cases = dict()
        for i in range(len(self.data_source)):
            idx, _, _, _, _, _ = np.unravel_index(i, self.data_source.shape)
            case_id = self.slides_frame.iloc[idx, CASE_INDEX]
            if case_id in self.cases:
                self.cases[case_id].append(i)
            else:
                self.cases[case_id] = [i]

    def __iter__(self):
        num_samples = 0
        max_samples = float("inf") if self.num_samples is None else self.num_samples
        for case in self.cases.keys():
            for sample in random.sample(self.cases[case], self.patches_per_case):
                if num_samples >= max_samples:
                    return
                num_samples += 1
                yield sample
    
    def __len__(self):
        return len(self.cases.keys()) * self.batch_size