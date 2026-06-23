import torch
import torch.nn as nn


class SoftHistogram(nn.Module):
    """Differentiable per-image, per-channel histogram for images in [0, 1]."""

    def __init__(
        self,
        bins=64,
        vmin=0.0,
        vmax=1.0,
        sigma=0.02,
        eps=1e-8,
        chunk_size=65536,
    ):
        super().__init__()
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.bins = bins
        self.vmin = vmin
        self.vmax = vmax
        self.sigma = sigma
        self.eps = eps
        self.chunk_size = chunk_size
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"expected a BCHW tensor, got shape {tuple(x.shape)}")

        batch_size, channels, _, _ = x.shape
        pixels = x.clamp(self.vmin, self.vmax).reshape(batch_size, channels, -1)
        centers = self.centers.to(dtype=x.dtype).view(1, 1, 1, self.bins)

        # Process pixels in chunks to avoid allocating a B x C x H*W x bins
        # tensor for a full-resolution image. The operations remain differentiable.
        histogram = x.new_zeros(batch_size, channels, self.bins)
        num_pixels = pixels.shape[-1]
        chunk_size = self.chunk_size or num_pixels
        for start in range(0, num_pixels, chunk_size):
            pixel_chunk = pixels[..., start : start + chunk_size].unsqueeze(-1)
            weights = torch.exp(
                -0.5 * ((pixel_chunk - centers) / (self.sigma + 1e-12)).square()
            )
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            histogram = histogram + weights.sum(dim=2)

        return histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(self.eps)


class ColorHistogramKLLoss(nn.Module):
    """Color consistency loss from SIED.

    The arguments are ordered as ``(output, ground_truth)``. For every image and
    RGB channel, this implements

        sum_c H_out(c) * log(H_out(c) / (H_gt(c) + tau)).

    The result is summed over channels and histogram bins, then averaged over
    the batch.
    """

    def __init__(self, num_bins=64, sigma=0.02, tau=1e-8, chunk_size=65536):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")

        self.num_bins = num_bins
        self.tau = tau
        self.compute_histogram = SoftHistogram(
            bins=num_bins,
            sigma=sigma,
            eps=tau,
            chunk_size=chunk_size,
        )

    def forward(self, img_out, img_gt):
        if img_out.shape != img_gt.shape:
            raise ValueError(
                "output and ground-truth images must have the same shape, "
                f"got {tuple(img_out.shape)} and {tuple(img_gt.shape)}"
            )

        hist_out = self.compute_histogram(img_out)
        hist_gt = self.compute_histogram(img_gt)

        # Paper direction: D_KL(H_out || H_gt). tau is added to the
        # denominator exactly as in the paper equation.
        loss_per_bin = hist_out * (
            torch.log(hist_out.clamp_min(self.tau))
            - torch.log(hist_gt + self.tau)
        )
        return loss_per_bin.sum(dim=(-1, -2)).mean()
