import numpy as np


class TimeSeriesTokeniser:
    """Minimal equal-width-bin tokeniser for 1D time series data."""

    def __init__(self, num_bins=256, clip=True, eps=1e-8):
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")

        self.num_bins = int(num_bins)
        self.clip = clip
        self.eps = eps

        self.mean_ = None
        self.std_ = None
        self.bin_edges_ = None
        self.bin_centres_ = None

    def fit(self, values):
        """Learn normalisation stats and regular bin edges from data."""
        values = self._as_1d_float_array(values)

        self.mean_ = float(values.mean())
        self.std_ = float(values.std())
        if self.std_ < self.eps:
            self.std_ = 1.0

        normalised = self.normalise(values)
        low = float(normalised.min())
        high = float(normalised.max())

        if abs(high - low) < self.eps:
            low -= 0.5
            high += 0.5

        self.bin_edges_ = np.linspace(low, high, self.num_bins + 1)
        self.bin_centres_ = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0
        return self

    def normalise(self, values):
        """Convert raw values to z-score normalised values."""
        self._check_normalisation_fitted()
        values = self._as_1d_float_array(values)
        return (values - self.mean_) / self.std_

    def denormalise(self, values):
        """Convert z-score normalised values back to raw scale."""
        self._check_normalisation_fitted()
        values = self._as_1d_float_array(values)
        return values * self.std_ + self.mean_

    def quantise(self, normalised_values):
        """Map normalised values to integer token IDs."""
        self._check_bins_fitted()
        normalised_values = self._as_1d_float_array(normalised_values)

        if self.clip:
            normalised_values = np.clip(
                normalised_values,
                self.bin_edges_[0],
                self.bin_edges_[-1],
            )

        tokens = np.digitize(normalised_values, self.bin_edges_[1:-1])
        return tokens.astype(np.int64)

    def dequantise(self, tokens):
        """Map token IDs back to normalised bin-centre values."""
        self._check_bins_fitted()
        tokens = np.asarray(tokens, dtype=np.int64)

        if np.any(tokens < 0) or np.any(tokens >= self.num_bins):
            raise ValueError(f"tokens must be in range [0, {self.num_bins - 1}]")

        return self.bin_centres_[tokens]

    def encode(self, values):
        """Normalise raw values and quantise them into token IDs."""
        return self.quantise(self.normalise(values))

    def decode(self, tokens):
        """Convert token IDs back to approximate raw values."""
        return self.denormalise(self.dequantise(tokens))

    def fit_encode(self, values):
        """Fit the tokeniser and return token IDs for the same values."""
        return self.fit(values).encode(values)

    def state_dict(self):
        """Return a serialisable snapshot of the fitted tokeniser."""
        self._check_bins_fitted()
        return {
            "num_bins": self.num_bins,
            "clip": self.clip,
            "eps": self.eps,
            "mean": self.mean_,
            "std": self.std_,
            "bin_edges": self.bin_edges_.copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild a fitted tokeniser from state_dict output."""
        tokeniser = cls(
            num_bins=state["num_bins"],
            clip=state.get("clip", True),
            eps=state.get("eps", 1e-8),
        )
        tokeniser.mean_ = float(state["mean"])
        tokeniser.std_ = float(state["std"])
        tokeniser.bin_edges_ = np.asarray(state["bin_edges"], dtype=np.float64)
        tokeniser.bin_centres_ = (
            tokeniser.bin_edges_[:-1] + tokeniser.bin_edges_[1:]
        ) / 2.0
        return tokeniser

    def _check_normalisation_fitted(self):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit must be called before normalising values")

    def _check_bins_fitted(self):
        self._check_normalisation_fitted()
        if self.bin_edges_ is None or self.bin_centres_ is None:
            raise RuntimeError("fit must be called before quantising values")

    @staticmethod
    def _as_1d_float_array(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        if values.size == 0:
            raise ValueError("values must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        return values
