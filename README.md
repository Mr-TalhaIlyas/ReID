## Tips

The error "all query identities do not appear in gallery" typically occurs when there is no overlap between the person identities (pids) in your query set and your gallery set. In other words, for each query image, there is no image in the gallery set from the same person but from a different camera view.

In the code provided, the same dataset is used for the training, query, and gallery sets. However, in a real-world scenario, you would typically want to split your data into separate training, query, and gallery sets. 

The key point is that for each person identity, there should be images from different camera views in the query and gallery sets. This is because the purpose of person re-identification is to match images of the same person across different camera views.

Here's a simple way to split your data into separate query and gallery sets, ensuring that for each person, there are images from different camera views in both sets:

```python
import os
import os.path as osp
import glob
import re
import random

from torchreid.data.datasets import ImageDataset

class NewDataset(ImageDataset):
    dataset_dir = 'new_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        # Get image list
        image_list = glob.glob(osp.join(self.data_dir, '*.png'))
        
        # Process image list
        dataset = []
        pid_container = set()
        for img_path in image_list:
            img_name = osp.basename(img_path)
            pid, camid = map(int, re.findall(r'\d+', img_name)[1:3]) # Parse pid and camid from filename
            pid_container.add(pid)
            dataset.append((img_path, pid, camid))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Re-assign labels to pid in dataset
        dataset = [(img_path, pid2label[pid], camid) for img_path, pid, camid in dataset]

        random.shuffle(dataset)  # Shuffle the dataset

        split_point = int(len(dataset) * 0.5)  # Split the dataset into two halves

        query = dataset[:split_point]  # Use the first half as the query set
        gallery = dataset[split_point:]  # Use the second half as the gallery set

        # Ensure that for each person, there are images from different camera views in both sets
        for img_path, pid, camid in query:
            if any(item[1] == pid and item[2] != camid for item in gallery):
                continue  # This person has images from different camera views in both sets
            else:
                # Move one image of this person from the query set to the gallery set
                for item in query:
                    if item[1] == pid:
                        query.remove(item)
                        gallery.append(item)
                        break

        super(NewDataset, self).__init__(dataset, query, gallery, **kwargs)


# Register the dataset
from torchreid.data import DataManager

torchreid.data.register_image_dataset('new_dataset', NewDataset)

# Use the dataset
datamanager = DataManager(
    root='path_to_your_data',
    sources='new_dataset',
    targets='new_dataset',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

```

This code first shuffles the dataset and splits it into two halves. It then checks for each person in the query set, if there are images of this person from a different camera view in the gallery set. If not, it moves one image of this person from the query set to the gallery set. This ensures that for each person, there are images from different camera views in both the query and gallery sets.

# References

link1:  https://kaiyangzhou.github.io/deep-person-reid/user_guide.html#prepare-datasets
link2: https://github.com/KaiyangZhou/deep-person-reid/issues/207
