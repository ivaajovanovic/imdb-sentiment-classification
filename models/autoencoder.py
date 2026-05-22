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

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderNet(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TfidfAutoencoder(BaseAutoencoder):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        n_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder = _EncoderNet(input_dim, hidden_dim, latent_dim).to(self.device)
        self._decoder = _DecoderNet(latent_dim, hidden_dim, input_dim).to(self.device)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=learning_rate,
        )

        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X_train, X_val) -> dict:
       
        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        t0 = time.perf_counter()

        for epoch in range(self.n_epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss   = self._run_epoch(val_loader,   training=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

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

    #za mlflow
    def get_params(self) -> dict:
        return {
            "input_dim":     self.input_dim,
            "hidden_dim":    self.hidden_dim,
            "latent_dim":    self.latent_dim,
            "n_epochs":      self.n_epochs,
            "batch_size":    self.batch_size,
            "learning_rate": self.learning_rate,
            "device":        self.device,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def training_time(self) -> float:
        """Wall-clock training time in seconds."""
        return self._training_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X, shuffle: bool) -> DataLoader:
        """Wrap X in a TfidfDataset and return a DataLoader."""
        dataset = _TfidfDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        if training:
            self._encoder.train()
            self._decoder.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                reconstructed = self._decoder(self._encoder(batch))
                loss = self._criterion(reconstructed, batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
        else:
            self._encoder.eval()
            self._decoder.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    reconstructed = self._decoder(self._encoder(batch))
                    loss = self._criterion(reconstructed, batch)
                    total_loss += loss.item()

        return total_loss / len(loader)

    def _to_tensor(self, X) -> torch.Tensor:
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return torch.FloatTensor(np.array(X)).to(self.device)

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train(X_train, X_val) first."
            )