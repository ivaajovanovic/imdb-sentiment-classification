import time
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.base_autoencoder import BaseAutoencoder


class _TfidfAugment:
    """
    Augmentacija TF-IDF vektora za SupContrast.

    Kreira dva različita "pogleda" iste recenzije:
    - View 1: random dropout nenultih featuredova (feature_drop_rate)
    - View 2: isti dropout ali sa drugačijim random seed-om

    Ovo simulira da su neke reči "izostavljene" iz recenzije —
    isti sentiment, drugačiji podskup reči.
    """

    def __init__(self, feature_drop_rate: float = 0.1) -> None:
        self.feature_drop_rate = feature_drop_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Primeni dropout samo na nenulte pozicije."""
        if self.feature_drop_rate == 0.0:
            return x
        mask = (x > 0).float()
        drop = torch.bernoulli(
            torch.full_like(x, 1.0 - self.feature_drop_rate)
        )
        return x * mask * drop


class _TfidfDataset(Dataset):
    """
    Dataset koji vraća (view1, view2, label) za SupContrast treniranje,
    ili samo (x, label) za val/test.
    """

    def __init__(self, X_sparse, y=None, augment: bool = False,
                 feature_drop_rate: float = 0.1) -> None:
        self.X = X_sparse
        self.y = y
        self.augment = augment
        self.aug = _TfidfAugment(feature_drop_rate)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        if scipy.sparse.issparse(row):
            row = row.toarray().squeeze()
        x = torch.FloatTensor(row)

        if self.augment:
            # Dva različita augmentovana pogleda iste recenzije
            view1 = self.aug(x.clone())
            view2 = self.aug(x.clone())
            if self.y is not None:
                return view1, view2, x, torch.tensor(self.y[idx], dtype=torch.long)
            return view1, view2, x
        else:
            if self.y is not None:
                return x, torch.tensor(self.y[idx], dtype=torch.long)
            return x


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


class TfidfAutoencoder(BaseAutoencoder):
    """
    Autoencoder sa pravim SupContrast pristupom.

    Za svaku recenziju kreira dva augmentovana pogleda (dropout featuredova).
    Supervised Contrastive Loss prima features shape [bsz, n_views, dim]
    kao u originalnom SupContrast paperu.

    total_loss = reconstruction_loss + contrastive_weight * contrastive_loss
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
        feature_drop_rate: float = 0.1,
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
        self.temperature = temperature
        self.feature_drop_rate = feature_drop_rate  # koliko featuredova dropout-ujemo
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
        # Train loader sa augmentacijom, val loader bez
        train_loader = self._make_loader(X_train, y_train, shuffle=True, augment=True)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False, augment=False)

        history: dict = {
            "train_loss":        [],
            "val_loss":          [],
            "train_recon":       [],
            "train_contrastive": [],
            "batch_losses":      [],
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
            "feature_drop_rate":  self.feature_drop_rate,
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
        Supervised Contrastive Loss — SupContrast stil.

        Parametri
        ---------
        features : (bsz, n_views, latent_dim)
            Latentne reprezentacije za svaki pogled.
            n_views=2: [view1_encodings, view2_encodings]
        labels : (bsz,)
            Labele klasa (0 ili 1).

        Returns
        -------
        torch.Tensor
            Skalarni loss.
        """
        bsz, n_views, dim = features.shape

        # Spoji sve poglede: (bsz * n_views, dim)
        # Redosled: [r1_v1, r1_v2, r2_v1, r2_v2, ...]
        features_flat = features.reshape(bsz * n_views, dim)
        features_flat = F.normalize(features_flat, dim=1)

        # Prošireni labels: svaki label se ponavlja n_views puta
        # [l1, l1, l2, l2, l3, l3, ...]
        labels_exp = labels.repeat_interleave(n_views)

        # Similarity matrica (bsz*n_views x bsz*n_views)
        sim_matrix = torch.matmul(features_flat, features_flat.T) / self.temperature

        # Maska istih klasa
        labels_col = labels_exp.unsqueeze(1)
        same_class_mask = (labels_col == labels_col.T).float()

        # Ukloni dijagonalu
        total = bsz * n_views
        identity = torch.eye(total, device=self.device)
        same_class_mask = same_class_mask * (1 - identity)

        # Numerički stabilno
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix) * (1 - identity)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_positives = same_class_mask.sum(dim=1)
        loss = -(same_class_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

        return loss.mean()

    def _weighted_mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask    = (target > 0).float()
        weights = 1.0 + (self.nonzero_weight - 1.0) * mask
        return (weights * (output - target) ** 2).mean()

    def _make_loader(self, X, y, shuffle: bool, augment: bool = False) -> DataLoader:
        if y is not None:
            y_int = np.array([1 if label == 'positive' else 0 for label in y])
        else:
            y_int = None
        dataset = _TfidfDataset(
            X, y_int,
            augment=augment,
            feature_drop_rate=self.feature_drop_rate,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader, training: bool, epoch: int) -> tuple[dict, list]:
        total_loss        = 0.0
        total_recon       = 0.0
        total_contrastive = 0.0
        batch_losses      = []

        if training:
            self._encoder.train()
            self._decoder.train()
        else:
            self._encoder.eval()
            self._decoder.eval()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, batch_data in enumerate(loader):

                if training:
                    # Augmentovani batch: (view1, view2, original, labels) ili (view1, view2, original)
                    if len(batch_data) == 4:
                        view1, view2, x_orig, y_batch = batch_data
                        y_batch = y_batch.to(self.device)
                    else:
                        view1, view2, x_orig = batch_data
                        y_batch = None

                    view1  = view1.to(self.device)
                    view2  = view2.to(self.device)
                    x_orig = x_orig.to(self.device)

                    # Enkoduj oba pogleda
                    z1 = self._encoder(view1)  # (bsz, latent_dim)
                    z2 = self._encoder(view2)  # (bsz, latent_dim)

                    # Reconstruction na originalnom x (ne augmentovanom)
                    z_orig        = self._encoder(x_orig)
                    reconstructed = self._decoder(z_orig)
                    recon_loss    = self._weighted_mse(reconstructed, x_orig)

                    # Contrastive loss sa oba pogleda: (bsz, 2, latent_dim)
                    if y_batch is not None and self.contrastive_weight > 0:
                        features = torch.stack([z1, z2], dim=1)
                        cont_loss = self._supervised_contrastive_loss(features, y_batch)
                        loss = recon_loss + self.contrastive_weight * cont_loss
                    else:
                        cont_loss = torch.tensor(0.0)
                        loss = recon_loss

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                else:
                    # Val batch: (x, labels) ili samo x
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        x_batch, y_batch = batch_data
                        y_batch = y_batch.to(self.device)
                    else:
                        x_batch = batch_data
                        y_batch = None

                    x_batch       = x_batch.to(self.device)
                    z             = self._encoder(x_batch)
                    reconstructed = self._decoder(z)
                    recon_loss    = self._weighted_mse(reconstructed, x_batch)
                    cont_loss     = torch.tensor(0.0)
                    loss          = recon_loss

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