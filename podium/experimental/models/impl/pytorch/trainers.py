import time

import torch

from podium.experimental.models.trainer import AbstractTrainer


class TorchTrainer(AbstractTrainer):
    def __init__(self, num_epochs, device, iterator, valid_data=None):
        self.epochs = num_epochs
        self.valid_data = valid_data
        self.device = device
        self.iterator = iterator

    def train(self, model, dataset, feature_transformer, label_transform_fun, **kwargs):

        for _ in range(self.epochs):
            total_time = time.time()
            for batch, (batch_x, batch_y) in enumerate(self.iterator(dataset)):
                t = time.time()
                X = feature_transformer.transform(batch_x).swapaxes(0, 1)  # swap B and T
                X = torch.from_numpy(X).to(self.device)
                y = torch.from_numpy(label_transform_fun(batch_y)).to(self.device)

                return_dict = model.fit(X, y)

                print(
                    "[Batch]: {} in {:.5f} seconds, loss={:.5f}".format(
                        batch, time.time() - t, return_dict["loss"]
                    ),
                    end="\r",
                    flush=True,
                )

            print(f"\nTotal time for train epoch: {time.time() - total_time}")

            total_time = time.time()
            for batch, (batch_x, batch_y) in enumerate(self.iterator(self.valid_data)):
                t = time.time()
                X = feature_transformer.transform(batch_x).swapaxes(0, 1)  # swap B and T
                X = torch.from_numpy(X).to(self.device)
                y = torch.from_numpy(label_transform_fun(batch_y)).to(self.device)

                return_dict = model.evaluate(X, y)
                loss = return_dict["loss"]
                print(
                    "[Valid]: {} in {:.5f} seconds, loss={:.5f}".format(
                        batch, time.time() - t, loss
                    ),
                    end="\r",
                    flush=True,
                )

            print(f"\nTotal time for valid epoch: {time.time() - total_time}")
