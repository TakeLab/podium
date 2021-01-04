import torch

from podium.experimental.models import AbstractSupervisedModel


class TorchModel(AbstractSupervisedModel):
    def __init__(
        self,
        model_class,
        criterion,
        optimizer,
        device=torch.device("cpu"),
        **model_config,
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.device = device
        self.optimizer_class = optimizer

        self._model = model_class(**model_config).to(self.device)
        self.optimizer = optimizer(self.model.parameters(), model_config["lr"])

        self.criterion = criterion

    @property
    def model(self):
        return self._model

    def __call__(self, X):
        """
        Call the forward method of the internalized model.
        """
        return self.model(X)

    def fit(self, X, y, **kwargs):
        """
        Fit the model on (X, y).

        Assumes that the model is in training mode.
        """
        # Train-specific boilerplate code
        self.model.train()
        self.model.zero_grad()

        return_dict = self(X)
        logits = return_dict["pred"].view(-1, self.model_config["num_classes"])

        loss = self.criterion(logits, y.squeeze())
        return_dict["loss"] = loss

        # Optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config["clip"])
        self.optimizer.step()
        return return_dict

    def predict(self, X, return_as_numpy=True, **kwargs):
        """
        Return the outputs of the model for inputs X.
        """
        self.model.eval()
        with torch.no_grad():
            return_dict = self(X)

            if return_as_numpy:
                # Cast everything to numpy
                preds = return_dict["pred"]
                # .cpu() is a no-op if the model is already on cpu
                preds = preds.cpu().numpy()
                return_dict["pred"] = preds

            return return_dict

    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model (compute loss) on (X, y).

        Assumes that the model is in evaluation mode.
        """

        self.model.eval()
        with torch.no_grad():
            return_dict = self(X)
            logits = return_dict["pred"].view(-1, self.model_config["num_classes"])
            loss = self.criterion(logits, y.squeeze())
            return_dict["loss"] = loss
            return return_dict

    def reset(self, **kwargs):
        """
        Reset (re-initialize) the model.

        Also resets the internal state of the optimizer.
        """
        self._model = self.model_class(self.model_config).to(self.model_config["device"])
        self.optimizer = self.optimizer_class(
            self.model.parameters(), self.model_config["lr"]
        )

    def __setstate__(self, state):
        self.model_class = state["model_class"]
        self.model_config = state["model_config"]
        self.device = state["device"]

        # Deserialize model
        model = self.model_class(**self.model_config)
        model.load_state_dict(state["model_state"])
        self._model = model.to(self.device)

        # Deserialize optimizer
        self.optimizer_class = state["optimizer_class"]
        self.optimizer = self.optimizer_class(
            self.model.parameters(), self.model_config["lr"]
        )
        self.optimizer.load_state_dict(state["optimizer_state"])

        # Deserialize loss
        loss_class = state["loss_class"]
        self.criterion = loss_class()
        self.criterion.load_state_dict(state["loss_state"])

    def __getstate__(self):
        state = {
            "model_class": self.model_class,
            "model_config": self.model_config,
            "model_state": self.model.state_dict(),
            "optimizer_class": self.optimizer_class,
            "optimizer_state": self.optimizer.state_dict(),
            "loss_class": self.criterion.__class__,
            "loss_state": self.criterion.state_dict(),
            "device": self.device,
        }
        return state
