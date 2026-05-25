import time
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.base_autoencoder import BaseAutoencoder


class _TfidfDataset(Dataset):

    def __init__(self, X_sparse, y=None) -> None:
        self.X = X_sparse
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        if scipy.sparse.issparse(row):
            row = row.toarray().squeeze()
        x_tensor = torch.FloatTensor(row)
        if self.y is not None:
            return x_tensor, torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor


class _EncoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TfidfAutoencoder(BaseAutoencoder):
    """
    Autoencoder sa Supervised Contrastive Loss.

    Kombinuje dva lossa:
    - Reconstruction loss (weighted MSE) — uči opšte reprezentacije
    - Supervised Contrastive loss — gura iste klase bliže, različite dalje

    total_loss = reconstruction_loss + contrastive_weight * contrastive_loss

    Supervised Contrastive loss:
    - Za svaki primer u batchu, privlači sve primere iste klase
    - Odbija sve primere različite klase
    - Direktno optimizuje organizaciju latentnog prostora po sentimentu
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        n_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        nonzero_weight: float = 10.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        log_every: int = 50,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nonzero_weight = nonzero_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature  # kontroliše "oštrinu" separacije
        self.log_every = log_every
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder = _EncoderNet(input_dim, hidden_dim, latent_dim).to(self.device)
        self._decoder = _DecoderNet(latent_dim, hidden_dim, input_dim).to(self.device)

        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) +
            list(self._decoder.parameters()),
            lr=learning_rate,
        )

        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X_train, X_val, y_train=None, y_val=None) -> dict:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

        history: dict = {
            "train_loss":       [],
            "val_loss":         [],
            "train_recon":      [],
            "train_contrastive":[],
            "batch_losses":     [],
        }

        t0 = time.perf_counter()

        for epoch in range(self.n_epochs):
            train_metrics, batch_losses = self._run_epoch(
                train_loader, training=True, epoch=epoch
            )
            val_metrics, _ = self._run_epoch(val_loader, training=False, epoch=epoch)

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["train_recon"].append(train_metrics["recon"])
            history["train_contrastive"].append(train_metrics["contrastive"])
            history["batch_losses"].extend(batch_losses)

            print(
                f"Epoch [{epoch + 1:>3}/{self.n_epochs}]  "
                f"train={train_metrics['total']:.6f}  "
                f"val={val_metrics['total']:.6f}  "
                f"recon={train_metrics['recon']:.6f}  "
                f"contrastive={train_metrics['contrastive']:.6f}"
            )

        self._training_time = time.perf_counter() - t0
        self._is_trained = True
        return history

    def encode(self, X, batch_size: int = 256) -> np.ndarray:
        self._assert_trained()
        self._encoder.eval()
        results = []
        n = X.shape[0]
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tensor = self._to_tensor(X[start:end])
                results.append(self._encoder(tensor).cpu().numpy())
        return np.vstack(results)

    def reconstruct(self, X, batch_size: int = 256) -> np.ndarray:
        self._assert_trained()
        self._encoder.eval()
        self._decoder.eval()
        results = []
        n = X.shape[0]
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tensor = self._to_tensor(X[start:end])
                results.append(self._decoder(self._encoder(tensor)).cpu().numpy())
        return np.vstack(results)

    def get_params(self) -> dict:
        return {
            "input_dim":          self.input_dim,
            "hidden_dim":         self.hidden_dim,
            "latent_dim":         self.latent_dim,
            "n_epochs":           self.n_epochs,
            "batch_size":         self.batch_size,
            "learning_rate":      self.learning_rate,
            "nonzero_weight":     self.nonzero_weight,
            "contrastive_weight": self.contrastive_weight,
            "temperature":        self.temperature,
            "log_every":          self.log_every,
            "device":             self.device,
        }

    @property
    def training_time(self) -> float:
        return self._training_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss.

        Za svaki primer u batchu:
        - Privlači sve primere iste klase (positives)
        - Odbija sve primere različite klase (negatives)

        Parametri
        ---------
        features : (batch_size, latent_dim)
            L2-normalizovane latentne reprezentacije.
        labels : (batch_size,)
            Labele klasa (0 ili 1).

        Returns
        -------
        torch.Tensor
            Skalarni loss.
        """
        # L2 normalizacija — cosine similarity između svih parova
        features = F.normalize(features, dim=1)

        # Similarity matrica (batch_size x batch_size)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Maska — koji primeri su iste klase
        labels = labels.unsqueeze(1)
        same_class_mask = (labels == labels.T).float()

        # Ukloni dijagonalu (self-similarity)
        batch_size = features.shape[0]
        identity = torch.eye(batch_size, device=self.device)
        same_class_mask = same_class_mask * (1 - identity)

        # Numerički stabilno: oduzmi max pre exp
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix) * (1 - identity)

        # Log-sum za svaki primer
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Prosečan log prob za positive parove
        n_positives = same_class_mask.sum(dim=1)
        loss = -(same_class_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

        return loss.mean()

    def _weighted_mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask    = (target > 0).float()
        weights = 1.0 + (self.nonzero_weight - 1.0) * mask
        return (weights * (output - target) ** 2).mean()

    def _make_loader(self, X, y, shuffle: bool) -> DataLoader:
        if y is not None:
            y_int = np.array([1 if label == 'positive' else 0 for label in y])
        else:
            y_int = None
        dataset = _TfidfDataset(X, y_int)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader, training: bool, epoch: int) -> tuple[dict, list]:
        total_loss = 0.0
        total_recon = 0.0
        total_contrastive = 0.0
        batch_losses = []

        if training:
            self._encoder.train()
            self._decoder.train()
        else:
            self._encoder.eval()
            self._decoder.eval()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, batch_data in enumerate(loader):
                if isinstance(batch_data, (list, tuple)):
                    x_batch, y_batch = batch_data
                    y_batch = y_batch.to(self.device)
                else:
                    x_batch = batch_data
                    y_batch = None

                x_batch = x_batch.to(self.device)
                latent        = self._encoder(x_batch)
                reconstructed = self._decoder(latent)
                recon_loss    = self._weighted_mse(reconstructed, x_batch)

                if y_batch is not None and self.contrastive_weight > 0:
                    cont_loss = self._supervised_contrastive_loss(latent, y_batch)
                    loss = recon_loss + self.contrastive_weight * cont_loss
                else:
                    cont_loss = torch.tensor(0.0)
                    loss = recon_loss

                if training:
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                total_loss        += loss.item()
                total_recon       += recon_loss.item()
                total_contrastive += cont_loss.item()

                if training and batch_idx % self.log_every == 0:
                    batch_losses.append({
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "loss":  round(loss.item(), 6),
                    })
                    print(
                        f"  Batch {batch_idx:>4}/{len(loader)}  "
                        f"loss={loss.item():.6f}"
                    )

        n = len(loader)
        return {
            "total":       total_loss        / n,
            "recon":       total_recon       / n,
            "contrastive": total_contrastive / n,
        }, batch_losses

    def _to_tensor(self, X) -> torch.Tensor:
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return torch.FloatTensor(np.array(X)).to(self.device)

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train(X_train, X_val) first."
            )