import time
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.base_autoencoder import BaseAutoencoder


class _TfidfDataset(Dataset):
    """Dataset koji čuva TF-IDF sparse matricu i labele."""

    def __init__(self, X_sparse, y=None) -> None:
        self.X = X_sparse
        self.y = y  # None za unsupervised, array za semi-supervised

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
    """Encoder: input_dim → hidden_dim → latent_dim sa Batch Normalization."""

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
    """Decoder: latent_dim → hidden_dim → input_dim sa Batch Normalization."""

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


class _ClassifierHead(nn.Module):
    """
    Mali klasifikator na vrhu encodera.
    latent_dim → 2 klase (negative/positive)
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TfidfAutoencoder(BaseAutoencoder):
    """
    Semi-supervised autoencoder za TF-IDF vektore.

    Kombinuje dva lossa:
    - Reconstruction loss (weighted MSE) — uči opšte reprezentacije
    - Classification loss (CrossEntropy) — gura sentiment klase razdvojeno

    total_loss = reconstruction_loss + classification_weight * classification_loss

    Parametri
    ---------
    classification_weight : float
        Koliko classification loss utiče u odnosu na reconstruction loss.
        0.0 = čisti unsupervised autoencoder
        1.0 = podjednako reconstruction i classification
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
        classification_weight: float = 1.0,
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
        self.classification_weight = classification_weight
        self.log_every = log_every
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder    = _EncoderNet(input_dim, hidden_dim, latent_dim).to(self.device)
        self._decoder    = _DecoderNet(latent_dim, hidden_dim, input_dim).to(self.device)
        self._classifier = _ClassifierHead(latent_dim).to(self.device)

        self._recon_criterion = nn.MSELoss(reduction='none')
        self._cls_criterion   = nn.CrossEntropyLoss()

        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) +
            list(self._decoder.parameters()) +
            list(self._classifier.parameters()),
            lr=learning_rate,
        )

        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X_train, X_val, y_train=None, y_val=None) -> dict:
        """
        Trenira autoencoder.

        Parametri
        ---------
        X_train, X_val : sparse matrix
            TF-IDF matrice.
        y_train, y_val : array of int, optional
            Labele (0=negative, 1=positive). Ako su None — čisti unsupervised.
        """
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

        history: dict = {
            "train_loss":      [],
            "val_loss":        [],
            "train_recon":     [],
            "train_cls":       [],
            "batch_losses":    [],
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
            history["train_cls"].append(train_metrics["cls"])
            history["batch_losses"].extend(batch_losses)

            print(
                f"Epoch [{epoch + 1:>3}/{self.n_epochs}]  "
                f"train_loss={train_metrics['total']:.6f}  "
                f"val_loss={val_metrics['total']:.6f}  "
                f"recon={train_metrics['recon']:.6f}  "
                f"cls={train_metrics['cls']:.6f}"
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
                batch = X[start:end]
                tensor = self._to_tensor(batch)
                latent = self._encoder(tensor)
                results.append(latent.cpu().numpy())
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
                batch = X[start:end]
                tensor = self._to_tensor(batch)
                reconstructed = self._decoder(self._encoder(tensor))
                results.append(reconstructed.cpu().numpy())
        return np.vstack(results)

    def get_params(self) -> dict:
        return {
            "input_dim":             self.input_dim,
            "hidden_dim":            self.hidden_dim,
            "latent_dim":            self.latent_dim,
            "n_epochs":              self.n_epochs,
            "batch_size":            self.batch_size,
            "learning_rate":         self.learning_rate,
            "nonzero_weight":        self.nonzero_weight,
            "classification_weight": self.classification_weight,
            "log_every":             self.log_every,
            "device":                self.device,
        }

    @property
    def training_time(self) -> float:
        return self._training_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask    = (target > 0).float()
        weights = 1.0 + (self.nonzero_weight - 1.0) * mask
        return (weights * (output - target) ** 2).mean()

    def _make_loader(self, X, y, shuffle: bool) -> DataLoader:
        # Konvertuj string labele u int ako su stringovi
        if y is not None:
            y_int = np.array([1 if label == 'positive' else 0 for label in y])
        else:
            y_int = None
        dataset = _TfidfDataset(X, y_int)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader, training: bool, epoch: int) -> tuple[dict, list]:
        total_loss = 0.0
        total_recon = 0.0
        total_cls = 0.0
        batch_losses = []

        if training:
            self._encoder.train()
            self._decoder.train()
            self._classifier.train()

            for batch_idx, batch_data in enumerate(loader):
                if isinstance(batch_data, (list, tuple)):
                    x_batch, y_batch = batch_data
                    y_batch = y_batch.to(self.device)
                else:
                    x_batch = batch_data
                    y_batch = None

                x_batch = x_batch.to(self.device)
                latent       = self._encoder(x_batch)
                reconstructed = self._decoder(latent)

                recon_loss = self._weighted_mse(reconstructed, x_batch)

                if y_batch is not None and self.classification_weight > 0:
                    logits    = self._classifier(latent)
                    cls_loss  = self._cls_criterion(logits, y_batch)
                    loss = recon_loss + self.classification_weight * cls_loss
                else:
                    cls_loss = torch.tensor(0.0)
                    loss = recon_loss

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss  += loss.item()
                total_recon += recon_loss.item()
                total_cls   += cls_loss.item()

                if batch_idx % self.log_every == 0:
                    batch_losses.append({
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "loss":  round(loss.item(), 6),
                    })
                    print(
                        f"  Batch {batch_idx:>4}/{len(loader)}  "
                        f"loss={loss.item():.6f}"
                    )
        else:
            self._encoder.eval()
            self._decoder.eval()
            self._classifier.eval()

            with torch.no_grad():
                for batch_data in loader:
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

                    if y_batch is not None and self.classification_weight > 0:
                        logits   = self._classifier(latent)
                        cls_loss = self._cls_criterion(logits, y_batch)
                        loss = recon_loss + self.classification_weight * cls_loss
                    else:
                        cls_loss = torch.tensor(0.0)
                        loss = recon_loss

                    total_loss  += loss.item()
                    total_recon += recon_loss.item()
                    total_cls   += cls_loss.item()

        n = len(loader)
        return {
            "total": total_loss  / n,
            "recon": total_recon / n,
            "cls":   total_cls   / n,
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