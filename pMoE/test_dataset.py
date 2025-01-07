from lib.utils import pMOEdataset, collate_fn_batching
from torch.utils.data import DataLoader

# Create a pMOEdataset object
dataset = pMOEdataset(dataset_name="squad", model_name="eastwind/tinymix-8x1b-chat")
dataset.prune_dataset(1024)

# Test dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: collate_fn_batching(batch, dataset.tokenizer))

for item in dataloader:
    print(item['input_ids'].shape)
    break


