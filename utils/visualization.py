import visdom
import numpy as np


class LineLogger(object):
    def __init__(self, xlabel, ylabel, vis_title, vis_legend, cls='iter'):
        self.cls = cls
        self.vis = visdom.Visdom()

        self.writer = self.create_vis_line(xlabel, ylabel, vis_title, vis_legend)

    def summarize(self, iteration, summaries_dict, update_type, epoch_size=1):
        self.update_vis_line(iteration, summaries_dict, update_type, epoch_size)

    def create_vis_line(self, xlabel, ylabel, title, legend):
        return self.vis.line(X=np.zeros((1, 3)),
                             Y=np.zeros((1, 3)),
                             opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend))

    def update_vis_line(self, iteration, summaries_dict, update_type, epoch_size=1):
        values = np.array([summaries_dict[key] for key in summaries_dict]).reshape((1, len(summaries_dict)))
        self.vis.line(X=np.ones((1, 3)) * iteration,
                      Y=values / epoch_size,
                      win=self.writer,
                      update=update_type)
        if iteration == 0 and self.cls == 'epoch':
            self.vis.line(X=np.ones((1, 3)),
                          Y=values,
                          win=self.writer,
                          update=update_type)


class ImageLogger(object):
    def __init__(self, size, title):
        self.vis = visdom.Visdom(port=8097)
        self.writer = self.create_vis_image(size, title)

    def create_vis_image(self, size, title):
        return self.vis.image(np.random.randn(*size),
                              opts=dict(title=title))

    def update_vis_image(self, image):
        self.vis.image(image, win=self.writer)
