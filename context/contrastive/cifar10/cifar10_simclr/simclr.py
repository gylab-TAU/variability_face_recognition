import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from utils import accuracy, save_checkpoint, save_config_file

torch.manual_seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        logits = logits / self.args.temperature
        return logits, labels

    def get_tsne_of_representations(self, test_dataloader, epoch):
        feature_extractor = nn.Sequential(*list(self.model.backbone.children())[:-1])
        feature_extractor.eval()
        self.model.eval()
        with torch.no_grad():
            features = []
            labels = []
            for batch_idx, (inputs, label) in enumerate(test_dataloader):
                inputs = inputs[0]
                inputs, label = inputs.to(device), label.to(device)
                features.append(feature_extractor(inputs).cpu().numpy())
                labels.append(label.cpu().numpy())

        features = np.concatenate(features, axis=0).squeeze(-1).squeeze(-1)
        labels = np.concatenate(labels, axis=0)

        # plot


        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(features)
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            legend="full",
            alpha=0.5
        )
        plt.show()
        # logger.report_matplotlib_figure(
        #     title="TSNE", series="plot", iteration=epoch, figure=plt, report_image=True
        # )
        return tsne_results, labels

    def train(self, train_loader,  val_loader, test_loader):

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        tsne_results, labels = self.get_tsne_of_representations(test_loader, epoch=-1)

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(device)

                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    print (f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}, Top5 accuracy: {top5[0]}")


                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            # get tsne of representations
            tsne_results, labels = self.get_tsne_of_representations(test_loader, epoch_counter)

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

