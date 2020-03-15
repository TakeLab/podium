import time
import torch

from takepod.models.trainer import AbstractTrainer

class TorchTrainer(AbstractTrainer):
    def __init__(self, num_epochs, device, iterator, valid_data=None):
        self.epochs = num_epochs
        self.valid_data = valid_data
        self.device = device
        self.iterator = iterator

    def train(self,
              model,
              dataset,
              feature_transformer,
              label_transform_fun,
              **kwargs):

        for _ in range(self.epochs):
            total_time = time.time()
            for batch_num, (batch_x, batch_y) in enumerate(self.iterator(dataset)):
                t = time.time()
                X = torch.from_numpy(
                    feature_transformer.transform(batch_x).swapaxes(0,1) # swap batch_size and T
                    ).to(self.device)
                y = torch.from_numpy(
                    label_transform_fun(batch_y)
                    ).to(self.device)

                return_dict = model.fit(X, y)

                print("[Batch]: {} in {:.5f} seconds, loss={:.5f}".format(
                       batch_num, time.time() - t, return_dict['loss']), 
                       end='\r', flush=True)

            print(f"\nTotal time for train epoch: {time.time() - total_time}")

            total_time = time.time()
            for batch_num, (batch_x, batch_y) in enumerate(self.iterator(self.valid_data)):
                t = time.time()
                X = torch.from_numpy(
                    feature_transformer.transform(batch_x).swapaxes(0,1) # swap batch_size and T
                    ).to(self.device)
                y = torch.from_numpy(
                    label_transform_fun(batch_y)
                    ).to(self.device)

                return_dict = model.evaluate(X, y)
                loss = return_dict['loss']
                print("[Valid]: {} in {:.5f} seconds, loss={:.5f}".format(
                       batch_num, time.time() - t, loss), 
                       end='\r', flush=True)

            print(f"\nTotal time for valid epoch: {time.time() - total_time}")
