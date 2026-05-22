import time
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.base_autoencoder import BaseAutoencoder


class _TfidfDataset(Dataset):
    def __init__(self, X_sparse) -> None:
        # batch po batch
        self.X = X_sparse

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.X[idx]
        if scipy.sparse.issparse(row):
            row = row.toarray().squeeze()
        return torch.FloatTensor(row)


class _EncoderNet(nn.Module):
    """Encoder: input_dim → hidden_dim → latent_dim sa Batch Normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # normalizuj pre ReLU
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # normalizuj pre ReLU
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
            nn.BatchNorm1d(hidden_dim),  # normalizuj pre ReLU
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # TF-IDF vrednosti između 0 i 1, bez BN ovde
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TfidfAutoencoder(BaseAutoencoder):
    """
    Autoencoder za TF-IDF vektore sa:
    - Batch Normalization za stabilno treniranje
    - Weighted MSE loss koji kažnjava greške na nenultim pozicijama
    - Per-batch loss logging na svakih log_every batcheva
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
        log_every: int = 50,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nonzero_weight = nonzero_weight  # težina za nenulte pozicije
        self.log_every = log_every            # loguj batch loss na svakih N batcheva
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder = _EncoderNet(input_dim, hidden_dim, latent_dim).to(self.device)
        self._decoder = _DecoderNet(latent_dim, hidden_dim, input_dim).to(self.device)
        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=learning_rate,
        )

        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X_train, X_val) -> dict:
        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   shuffle=False)

        history: dict = {
            "train_loss":   [],   # prosečan loss po epohi
            "val_loss":     [],   # prosečan loss po epohi
            "batch_losses": [],   # loss na svakih log_every batcheva
        }

        t0 = time.perf_counter()

        for epoch in range(self.n_epochs):
            train_loss, batch_losses = self._run_epoch(
                train_loader, training=True, epoch=epoch
            )
            val_loss, _ = self._run_epoch(val_loader, training=False, epoch=epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["batch_losses"].extend(batch_losses)

            print(
                f"Epoch [{epoch + 1:>3}/{self.n_epochs}]  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

        self._training_time = time.perf_counter() - t0
        self._is_trained = True
        return history

    def encode(self, X) -> np.ndarray:
        self._assert_trained()
        self._encoder.eval()
        tensor = self._to_tensor(X)
        with torch.no_grad():
            latent = self._encoder(tensor)
        return latent.cpu().numpy()

    def reconstruct(self, X) -> np.ndarray:
        self._assert_trained()
        self._encoder.eval()
        self._decoder.eval()
        tensor = self._to_tensor(X)
        with torch.no_grad():
            reconstructed = self._decoder(self._encoder(tensor))
        return reconstructed.cpu().numpy()

    def get_params(self) -> dict:
        return {
            "input_dim":      self.input_dim,
            "hidden_dim":     self.hidden_dim,
            "latent_dim":     self.latent_dim,
            "n_epochs":       self.n_epochs,
            "batch_size":     self.batch_size,
            "learning_rate":  self.learning_rate,
            "nonzero_weight": self.nonzero_weight,
            "log_every":      self.log_every,
            "device":         self.device,
        }

    @property
    def training_time(self) -> float:
        return self._training_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Weighted MSE — nenulte pozicije dobijaju nonzero_weight puta veću kaznu.

        Rešava problem sa sparse TF-IDF matricama gde obični MSE
        uči da outputuje sve nule jer to minimizuje loss.
        """
        mask = (target > 0).float()
        weights = 1.0 + (self.nonzero_weight - 1.0) * mask
        return (weights * (output - target) ** 2).mean()

    def _make_loader(self, X, shuffle: bool) -> DataLoader:
        dataset = _TfidfDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
    ) -> tuple[float, list]:
        """
        Jedna epoha treniranja ili evaluacije.

        Returns
        -------
        tuple
            (mean_loss, batch_losses) — batch_losses je prazan za val epohe.
        """
        total_loss = 0.0
        batch_losses = []

        if training:
            self._encoder.train()
            self._decoder.train()
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                reconstructed = self._decoder(self._encoder(batch))
                loss = self._weighted_mse(reconstructed, batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()

                # Loguj batch loss na svakih log_every batcheva
                if batch_idx % self.log_every == 0:
                    batch_losses.append({
                        "epoch":     epoch + 1,
                        "batch":     batch_idx,
                        "loss":      round(loss.item(), 6),
                    })
                    print(
                        f"  Batch {batch_idx:>4}/{len(loader)}  "
                        f"loss={loss.item():.6f}"
                    )
        else:
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    reconstructed = self._decoder(self._encoder(batch))
                    loss = self._weighted_mse(reconstructed, batch)
                    total_loss += loss.item()

        return total_loss / len(loader), batch_losses

    def _to_tensor(self, X) -> torch.Tensor:
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return torch.FloatTensor(np.array(X)).to(self.device)

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train(X_train, X_val) first."
            )