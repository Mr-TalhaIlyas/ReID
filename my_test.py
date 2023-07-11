#%%

'''
pip install --upgrade jupyter_client
pip install torch-tb-profiler
'''
import os
os.chdir('/home/user01/data/track/deep-person-reid/')
import torch
import torchreid
# torchreid.models.show_avai_models()
# %%
from torchreid import models, utils


# %%
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=32
)
#%%
# return train loader of source data
train_loader = datamanager.train_loader
batch = next(iter(train_loader))
# return test loader of target data
test_loader = datamanager.test_loader

# %%
model = models.build_model(name='osnet_x1_0',
                           num_classes=datamanager.num_train_pids,
                           loss='softmax')

torchreid.utils.load_pretrained_weights(model, '/home/user01/data/track/deep-person-reid/log/osnet_x1_0/model/model.pth.tar-60')

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
    save_dir='log/osnet_x1_0',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False,
    fixbase_epoch=5,
    open_layers='classifier'
)
# or open_layers=['fc', 'classifier'] if there is another fc layer that
# is randomly initialized, like resnet50_fc512
# %%
