import time
import torch

from utils.visualization import LineLogger
from dataset.config import voc as cfg
from ssd.utils_ssd.ops import adjust_learning_rate


class Trainer(object):
    def __init__(self, model, criterion, optimizer, data_loader, config, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.device = device

    def train(self):
        if self.config.resume:
            print("Resuming training, loading {}...".format(self.config.resume))
            self.model.load_weights(self.config.resume)
        else:
            print("Loading base network...")
            vgg_weights = torch.load(self.config.save_folder + self.config.basenet)
            self.model.vgg.load_state_dict(vgg_weights)

            print("Initializing added layers' weights...")
            self.model.init_extras_weights()

        print('Training SSD on', self.config.dataset, 'using the specified args:')
        print(self.config)

        # visualization init
        if self.config.visdom:
            vis_title = 'SSD.PyTorch on ' + self.config.dataset
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            self.iter_plot = LineLogger('Iteration', 'Loss', vis_title, vis_legend, cls='iter')
            self.epoch_plot = LineLogger('Epoch', 'Loss', vis_title, vis_legend, cls='epoch')

        self.batch_iterator = iter(self.data_loader)
        self.total_step = 0
        self.total_epoch = 0
        self.step_index = 0
        self.epochs = len(self.data_loader) // self.config.batch_size
        self.num_iter_per_epoch = len(self.data_loader) // self.epochs

        for cur_epoch in range(self.epochs):
            self.train_epoch()

        torch.save(self.model.state_dict(), self.config.save_folder + '' + self.config.dataset + '.pth')

    def train_epoch(self):
        loc_loss = 0
        conf_loss = 0
        for i in range(self.num_iter_per_epoch):
            loss_l, loss_c = self.train_step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

        if self.config.visdom:
            summaries_dict = {
                'loc': loc_loss,
                'conf': conf_loss,
                'loc+conf': loc_loss + conf_loss
            }
            # self.epoch_plot.summarize(self.total_epoch, summaries_dict, 'append', self.epochs)

        self.total_epoch += 1

    def train_step(self):
        if self.total_step in cfg['lr_steps']:
            self.step_index += 1
            adjust_learning_rate(self.config.lr, self.optimizer, self.config.gamma, self.step_index)

        images, targets = next(self.batch_iterator)
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]

        t0 = time.time()
        outputs = self.model(images)
        loss_l, loss_c = self.criterion(outputs, targets)
        loss = loss_l + loss_c

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t1 = time.time()
        self.total_step += 1

        if self.total_step % 10 == 0:
            print('Time: %.4f sec || iter: %d || Loss: %.4f ' % (t1 - t0, self.total_step, loss.item()))

        # visualization
        if self.config.visdom:
            summaries_dict = {
                'loc': loss_l.item(),
                'conf': loss_c.item(),
                'loc+conf': loss.item()
            }
            # self.iter_plot.summarize(self.total_step, summaries_dict, 'append')

        # save model
        if self.total_step != 0 and self.total_step % 5000 == 0:
            print("Save state, iter:", self.total_step)
            torch.save(self.model.state_dict(), 'weights/ssd300_VOC_' + str(self.total_step) + '.pth')

        return loss_l, loss_c
