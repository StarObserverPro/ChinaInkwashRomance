"""Learn a Chinese side-brush continuation from the approach to a given endpoint."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr, ndtri, softmax
import torch
from torch import nn
from torch.nn import functional as F

from frequency import band_masks, spectrum
from shared_brush import optical


PITCH = 1./32.
Y_AXIS = (np.arange(96)+.5)*PITCH-1.5
X_AXIS = (np.arange(128)+.5)*PITCH-2.75
Y, X = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
JOIN = -1.25
KNOWN = X < JOIN
RANGE = np.arange(np.count_nonzero(X_AXIS < JOIN), len(X_AXIS))
LIMITS = np.array((.035, .15, .4, .65, .88), dtype=np.float32)
TONES = np.array((.006, .072, .26, .52, .76, .95), dtype=np.float32)


@dataclass(frozen=True)
class Frame:
    end_x: float
    center_y: float
    width: float
    signed_tip: int = 0

    def capture(self, ink, *, visible_only=False):
        direction = self.signed_tip or 1
        x = self.end_x+self.width*X
        if visible_only:
            # Clamp before interpolation so the last known sample never blends
            # with an unavailable pixel on the other side of the join.
            last = np.floor(np.nextafter(self.end_x+self.width*JOIN, -np.inf))
            x = np.minimum(x, last)
        return map_coordinates(ink, (self.center_y+direction*self.width*Y, x), order=1, mode="constant").astype(np.float32)

    def project(self, field, shape):
        row, col = np.indices(shape, dtype=float)
        direction = self.signed_tip or 1
        y = (row-self.center_y)*direction/self.width
        x = (col-self.end_x)/self.width
        coordinates = ((y-Y_AXIS[0])/PITCH, (x-X_AXIS[0])/PITCH)
        return map_coordinates(field, coordinates, order=1, mode="constant"), x >= JOIN


def condition(ink, direction_known):
    approach = np.where(KNOWN, ink, 0.)
    rim = np.mean(approach[:, (X_AXIS >= JOIN-.58) & (X_AXIS < JOIN)], axis=1)
    return np.stack((approach, KNOWN, np.broadcast_to(rim[:, None], X.shape), X/3., Y/1.5,
                     np.broadcast_to(float(direction_known), X.shape))).astype(np.float32)


class EndPainter(nn.Module):
    def __init__(self):
        super().__init__()
        self.near = nn.Sequential(nn.Conv2d(6, 12, 5, padding=2), nn.SiLU(), nn.Conv2d(12, 12, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(nn.Conv2d(12, 20, 5, padding=2), nn.SiLU(),
                                     nn.Conv2d(20, 20, 3, dilation=3, padding=3), nn.SiLU(),
                                     nn.Conv2d(20, 20, 3, dilation=5, padding=5), nn.SiLU())
        self.paint = nn.Sequential(nn.Conv2d(32, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, len(TONES), 1))

    def forward(self, inputs):
        near = self.near(inputs)
        context = self.context(F.avg_pool2d(near, 4))
        context = F.interpolate(context, size=near.shape[-2:], mode="bilinear", align_corners=False)
        return self.paint(torch.cat((near, context), dim=1))


class InkLoss:
    """Score ink categories, dark support geometry, white continuity, and directional power."""

    def __init__(self):
        y, x = torch.from_numpy(Y[:, RANGE]).float(), torch.from_numpy(X[:, RANGE]).float()
        self.basis = torch.stack((torch.ones_like(x), x, y, x*x, x*y, y*y)).flatten(1)
        self.tones = torch.from_numpy(TONES)[None, :, None, None]
        self.limits = torch.from_numpy(LIMITS)
        shape = y.shape
        bands = band_masks(shape, wavelengths=((3, 6), (6, 12), (12, 24)))
        self.bands = torch.from_numpy(bands).flatten(1).float()
        self.window = torch.hann_window(shape[0])[:, None]*torch.hann_window(shape[1])

    def geometry(self, plane):
        return (plane.flatten(1) @ self.basis.T)*PITCH**2

    def edge(self, plane):
        dx = (plane[:, :, 1:]-plane[:, :, :-1]).abs().sum(dim=(-1, -2))*PITCH
        dy = (plane[:, 1:]-plane[:, :-1]).abs().sum(dim=(-1, -2))*PITCH
        width = (plane.sum(dim=(-1, -2))*PITCH**2).sqrt().clamp_min(.12)
        return torch.stack((dx/width, dy/width), dim=1)

    def white(self, probability, inside):
        weights = inside.sum(dim=(-1, -2)).clamp(min=1)
        p = probability*inside
        values = [p.sum(dim=(-1, -2))/weights]
        for offset in (2, 5, 9):
            values.extend(((p[:, :, :-offset]*p[:, :, offset:]).sum(dim=(-1, -2))/weights,
                           (p[:, :-offset, :]*p[:, offset:, :]).sum(dim=(-1, -2))/weights))
        return torch.stack(values, dim=1)

    def spectrum(self, plane):
        detail = plane-F.avg_pool2d(plane[:, None], 5, stride=1, padding=2)[:, 0]
        power = torch.fft.fft2(detail*self.window, norm="ortho").abs().square().flatten(1)
        return torch.log(power @ self.bands.T/plane.shape[-1]/plane.shape[-2]+1e-7)

    def __call__(self, network, inputs, ink, structure, resolved):
        logits, ink = network(inputs)[..., RANGE[0]:], ink[..., RANGE[0]:]
        target = torch.bucketize(ink.contiguous(), self.limits)
        probability = logits.softmax(dim=1)
        estimated = (probability*self.tones).sum(dim=1)
        weight = 1.+2.*(ink >= .15).float()
        gray = (weight*(estimated-ink).abs()).mean()
        likelihood = (weight*F.cross_entropy(logits, target, reduction="none")).mean()
        if not structure:
            return likelihood+.8*gray, (likelihood, gray)

        contact = (target >= 2).float()
        deep = (target >= 4).float()
        support, dark = probability[:, 2:].sum(dim=1), probability[:, 4:].sum(dim=1)
        shape = F.smooth_l1_loss(self.geometry(support), self.geometry(contact), beta=.14)
        shape += F.smooth_l1_loss(self.geometry(dark), self.geometry(deep), beta=.14)
        inside = contact.cumsum(dim=1).bool() & contact.flip(1).cumsum(dim=1).bool().flip(1)
        white = (self.white(probability[:, 0], inside)-self.white((target == 0).float(), inside)).abs().mean()
        power = (self.spectrum(estimated)-self.spectrum(ink)).abs().mean(dim=1)
        contour = (self.edge(support)-self.edge(contact)).abs().mean(dim=1)
        bands = (power*resolved).sum()/resolved.sum().clamp_min(1.)
        edge = (contour*resolved).sum()/resolved.sum().clamp_min(1.)
        return likelihood+.8*gray+2.*shape+.65*white+.12*bands+.04*edge, (likelihood, gray, shape, white, bands, edge)


@torch.inference_mode()
def probabilities(model, crop, direction_known):
    x = torch.from_numpy(condition(crop, direction_known)[None])
    return model(x)[0].softmax(dim=0).numpy()


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network = EndPainter().eval()
    network.load_state_dict(checkpoint["weights"])
    return network, tuple(checkpoint["sources"])


def propose(network, incoming, frame, *, blur=None, seed=None):
    """Propose only new contact. The caller assigns its optical dose and clock."""
    probability = probabilities(network, frame.capture(incoming, visible_only=True), bool(frame.signed_tip))
    ink, enabled = frame.project(render(probability, blur=blur, seed=seed), incoming.shape)
    return np.where(enabled, ink, 0.)


def correlations(items):
    """Estimate the length of residual ink correlation in the registered paper coordinates."""
    observed = []
    for prob, target in items:
        classes = np.searchsorted(LIMITS, target)
        cdf = np.cumsum(prob, axis=0)
        greater = np.take_along_axis(cdf, classes[None], axis=0)[0]
        p = np.take_along_axis(prob, classes[None], axis=0)[0]
        z = ndtri(np.clip(greater-.5*p, .005, .995))
        good = (prob[0] > .04) & (prob[0] < .96) & ~KNOWN
        rates = []
        for field, keep in ((z, good), (z.T, good.T)):
            select = keep[1:] & keep[:-1]
            first, second = field[:-1][select], field[1:][select]
            if len(first) < 20 or min(first.std(), second.std()) < 1e-6:
                rates.append(0.)
            else:
                rates.append(np.corrcoef(first, second)[0, 1])
        observed.append(rates)
    correlation = np.median(observed, axis=0).clip(.02, .99)
    return (1./np.sqrt(-4.*np.log(correlation))).tolist()


def render(probability, *, blur=None, seed=None):
    """Return an expected ink plane or one draw with learned spatial correlation."""
    if seed is None:
        return np.tensordot(TONES, probability, axes=1)
    rng = np.random.default_rng(seed)
    residual = gaussian_filter(rng.standard_normal(X.shape), blur)
    uniform = ndtr((residual-residual.mean())/max(1e-8, residual.std()))
    classes = np.sum(np.cumsum(probability, axis=0)[:-1] < uniform, axis=0)
    return TONES[classes]


def visible_signature(ink):
    visible = ink[:, :RANGE[0]]
    depth = optical(visible)
    center = np.sum(depth*Y[:, :RANGE[0]])/max(1e-8, depth.sum())
    mask = visible >= .15
    inside = np.maximum.accumulate(mask, axis=0) & np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    near = depth[:, -12:].mean()
    far = depth[:, :-12].mean()
    white = (visible <= .035) & inside
    return np.array([np.log(.02+near), np.log((.02+near)/(.02+far)), center, mask.mean(),
                     white.sum()/max(1, inside.sum()),
                     (white[:, :-3] & white[:, 3:]).sum()/max(1, inside.sum()),
                     (white[:-3] & white[3:]).sum()/max(1, inside.sum())])


def cap_signature(ink):
    cap = ink[:, RANGE[0]:]
    mask = cap >= .15
    depth = optical(cap)
    inside = np.maximum.accumulate(mask, axis=0) & np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    white = (cap <= .035) & inside
    moments = np.array([np.sum(depth*f)/max(1e-8, depth.sum()) for f in (X[:, RANGE[0]:], Y[:, RANGE[0]:])])
    power = np.log(spectrum(cap)["detail_bands"].ravel()+1e-7)
    return np.r_[mask.mean(), np.log(.02+depth.mean()), moments, white.sum()/max(1, inside.sum()),
                 (white[:, :-4] & white[:, 4:]).sum()/max(1, inside.sum()), power]


@dataclass(frozen=True)
class Transfer:
    capsules: np.ndarray
    normalized_inputs: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    log_prior: np.ndarray
    loss: tuple[float, float]

    def distribution(self, crop):
        sample = visible_signature(crop)/self.scales
        difference = np.square(self.normalized_inputs-sample)
        return softmax(self.log_prior-difference@self.weights)

    def sample(self, crop, *, seed=None):
        probability = self.distribution(crop)
        index = int(np.argmax(probability) if seed is None else np.random.default_rng(seed).choice(len(probability), p=probability))
        return self.capsules[index].copy(), index


def fit_transfer(crops, observed, courses):
    """Fit how to choose a whole ink continuation using only the visible prefix."""
    pictures = np.stack(crops)
    courses = np.asarray(courses)
    values = np.stack([visible_signature(im) for im in observed])
    query_scale = np.maximum(values.std(axis=0), [.12, .12, .08, .07, .04, .03, .03])
    values /= query_scale
    a = np.square(values[:, None]-values[None, :])
    future = np.stack([cap_signature(im) for im in pictures])
    future /= np.maximum(future.std(axis=0), .06)
    distance = np.square(future[:, None]-future[None, :]).mean(axis=2)
    counts = (courses[:, None] == courses[None, :]).sum(axis=1)
    prior = -np.log(counts)
    blocked = courses[:, None] == courses[None, :]
    eligible = np.where(blocked, -1e12, prior[None, :])
    expected = softmax(eligible-distance, axis=1)
    balancing = (1./counts)/(1./counts).sum()

    def objective(w):
        score = eligible-a@w
        log_prob = score-logsumexp(score, axis=1, keepdims=True)
        loss = -np.sum(balancing[:, None]*expected*log_prob)+.04*np.square(w).sum()
        error = balancing[:, None]*(expected-np.exp(log_prob))
        return float(loss), np.einsum("ij,ijk->k", error, a)+.08*w

    start = np.zeros(values.shape[-1])
    fit = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0., 4.)]*len(start))
    if not fit.success:
        raise ValueError(f"Course-held-out retrieval did not converge: {fit.message}")
    return Transfer(pictures, values, query_scale, fit.x, prior, (objective(start)[0], float(fit.fun)))
