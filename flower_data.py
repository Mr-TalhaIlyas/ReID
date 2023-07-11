#%%
import os
import os.path as osp
import glob
import re, random
import torchreid
from torchreid.data.datasets import ImageDataset
import matplotlib.pyplot as plt
from torchreid import models, utils

train_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/allv2/'
# test_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/all/'

train_data = 'train_flower'
# test_data = 'test_flower'

class NewDataset(ImageDataset):
    dataset_dir = train_dir

    def __init__(self, root='', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir
        print(self.data_dir)
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

        train = dataset
        query = dataset
        gallery = dataset

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

from torchreid.data import ImageDataManager

# Register the dataset
torchreid.data.register_image_dataset(train_data, NewDataset)

# Use the dataset
datamanager = ImageDataManager(
    root='reid-data',
    sources=train_data,
    # targets='new_dataset',
    height=128,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    # combineall=True,
    # market1501_500k=False,
    transforms=['random_flip', 'random_crop', 'color_jitter']
)

train_loader = datamanager.train_loader
batch = next(iter(train_loader))
idx = 30
print('CAMID',batch['camid'][idx])
print('person ID',batch['pid'][idx])
plt.imshow(batch['img'][idx].permute(1,2,0).detach().numpy())
# return test loader of target data
# test_loader = datamanager.test_loader

# %%
model = models.build_model(name='osnet_x1_0',
                           num_classes=datamanager.num_train_pids,
                           loss='triplet') #softmax
torchreid.utils.load_pretrained_weights(model, '/home/user01/data/track/deep-person-reid/log/osnet_x1_0/model/model.pth.tar-60')
model = model.cuda()

#%%
# New layer "classifier" has a learning rate of 0.01
# The base layers have a learning rate of 0.001
optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.001,
    # staged_lr=True,
    # new_layers='classifier',
    # base_lr_mult=0.1
)
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)
# %%
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model, optimizer, scheduler=scheduler
)
#%%
# print('Running test...')
# engine.run(
#     save_dir='/home/user01/data/track/deep-person-reid/log/flower_osnet_x1_0',
#     max_epoch=100,
#     eval_freq=10,
#     print_freq=10,
#     test_only=True,
#     fixbase_epoch=5,
#     open_layers='classifier'
# )
# print('Finished test...All clear')
#%%
engine.run(
    save_dir='/home/user01/data/track/deep-person-reid/log/only_train_flower_osnet_x1_0',
    max_epoch=100,
    eval_freq=101,#10
    print_freq=10,
    test_only=False,
    fixbase_epoch=5,
    open_layers='classifier'
)

# state = {
#     'state_dict': model.state_dict(),
#     'epoch': 64,
#     'rank1': 0.0,
#     'optimizer': optimizer.state_dict()
# }

# torchreid.utils.torchtools.save_checkpoint(state, save_dir='/home/user01/data/track/deep-person-reid/log/only_train_flower_osnet_x1_0', is_best=False, remove_module_from_keys=False)

























































#%%
train_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/allv2/'


# Get image list
image_list = glob.glob(osp.join(train_dir, '*.png'))

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










#%%
data_dir = train_dir
print(data_dir)
# Get image list
image_list = glob.glob(osp.join(data_dir, '*.png'))

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

train = dataset
query = dataset
gallery = dataset
