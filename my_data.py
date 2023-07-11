#%%
import os
os.chdir('/home/user01/data/track/deep-person-reid/')
import csv
import sys
import os
import os.path as osp
from fmutils import fmutils as fmu
import torchreid
from torchreid.data import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
from torchreid import models

train_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/all/'
test_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/all/'

train_data = 'train_flower'
test_data = 'test_flower'

#%%
class TrainDataset(ImageDataset):
    dataset_dir = train_dir
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.dataset_dir

        data = []
        data_dir = fmu.get_all_files(self.dataset_dir)

        for i in range(len(data_dir)):
            filename = Path(data_dir[i]).stem
            frame_number = filename.split('_')[0]
            pid = int(filename.split('_')[1]) - 1 # to make zero based
            camid = int(filename.split('_')[2])

            if camid == 1:
                pid = pid + 8 # beacuse in cannon camera max pid is 8 

            data.append((data_dir[i], pid, camid))

        query = data
        gallery = data

        super(TrainDataset, self).__init__(data, query, gallery, **kwargs)

class TestDataset(ImageDataset):
    dataset_dir = test_dir
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.dataset_dir

        data = []
        data_dir = fmu.get_all_files(self.dataset_dir)

        for i in range(len(data_dir)):
            filename = Path(data_dir[i]).stem
            frame_number = filename.split('_')[0]
            pid = int(filename.split('_')[1]) - 1 # to make zero based
            camid = int(filename.split('_')[2])

            if camid == 1:
                pid = pid + 8 # beacuse in cannon camera max pid is 8 

            data.append((data_dir[i], pid, camid))

        query = data
        gallery = data

        super(TestDataset, self).__init__(data, query, gallery, **kwargs)

#%%
torchreid.data.register_image_dataset(train_data, TrainDataset)
torchreid.data.register_image_dataset(test_data, TestDataset)
#%%
# use your own dataset only
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources=train_data,
    # targets=test_data,
    height=128,
    width=128,
    batch_size_train=32,
    batch_size_test=32
)
# return train loader of source data
# train_loader = datamanager.train_loader
# batch = next(iter(train_loader))
# test_loader = datamanager.test_loader

#%%
model = models.build_model(name='osnet_x1_0',
                           num_classes=datamanager.num_train_pids,
                           loss='softmax')

# torchreid.utils.load_pretrained_weights(model, '/home/user01/data/track/deep-person-reid/log/osnet_x1_0/model/model.pth.tar-60')

model = model.cuda()
#%% 
# New layer "classifier" has a learning rate of 0.01
# The base layers have a learning rate of 0.001
optimizer = torchreid.optim.build_optimizer(
    model,
    optim='sgd',
    lr=0.01,
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

engine.run(
    save_dir='log/flowers_osnet_x1_0',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False,
    fixbase_epoch=5,
    open_layers='classifier'
)
#%%




















































#%%
'''
Data sanity check
'''
train_dir = '/home/user01/data/track/deep-person-reid/reid-data/flower/test/'

data = []
data_dir = fmu.get_all_files(train_dir)
apid = []
for i in range(len(data_dir)):
    filename = Path(data_dir[i]).stem
    frame_number = filename.split('_')[0]
    pid = int(filename.split('_')[1]) - 1
    camid = int(filename.split('_')[2])

    if camid == 1:
        pid = pid + 8 # beacuse in cannon camera max pid is 8 
    apid.append(pid)
    data.append((data_dir[i], pid, camid))

query = data
gallery = data

x = np.asarray(apid)
print(np.unique(x))
# %%
