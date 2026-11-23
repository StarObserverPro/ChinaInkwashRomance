"""Local paper moisture and conservative mobility of recently deposited Chinese ink."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


SPREAD = np.array((0., .002, .009, .027, .060))
# Times are relative to a complete local recontact observation.
ABSORPTION = 2.0
BINDING = 1.2
WATER_TO_PIGMENT_SPREAD = 2.
DRY_LINK = .10
TRACE_CROSS = .22
ANGULAR_SCALE = 128./35.  # Reciprocal mean of cos(theta)**8 over all directions.
GRID = ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :]))
NEIGHBORS = (*GRID, (np.s_[:-1, :-1], np.s_[1:, 1:]), (np.s_[1:, :-1], np.s_[:-1, 1:]))
SCALAR_LINKS = tuple((a, b, 1.) for a, b in GRID)


def exchange(field, coefficient, gate=None, links=SCALAR_LINKS):
    """Pairwise no-flux transfer on a paper grid."""
    delta = np.zeros_like(field)
    for a, b, conductance in links:
        change = coefficient*conductance*(field[a]-field[b])
        if gate is not None:
            change *= np.maximum(gate[a], gate[b])
        delta[a] -= change
        delta[b] += change
    field += delta


def trace_guide(depth, width, cross, spacing=(1., 1.)):
    """Infer a local tangent, with `cross` mixing in undirected wet conductance."""
    radius = max(.55, .045*width)
    dx, dy = spacing
    sample = (radius/dy, radius/dx)
    smooth_ink = gaussian_filter(depth, sample)
    feature = np.log1p(smooth_ink/.10)
    gy, gx = np.gradient(feature, dy, dx)
    yy, xx, xy = (gaussian_filter(field, (1.6*sample[0], 1.6*sample[1])) for field in (gy*gy, gx*gx, gx*gy))
    # The tangent is perpendicular to the main local image gradient. A doubled
    # angle encodes that direction without choosing one of the two signs.
    trace_x = (yy-xx)/(xx+yy+1e-6)
    trace_y = -2.*xy/(xx+yy+1e-6)
    links = []
    for (a, b), (i, j) in zip(NEIGHBORS, ((dx, 0.), (0., dy), (dx, dy), (dx, -dy)), strict=True):
        length = i*i+j*j
        direction = ((i*i-j*j)*trace_x+2.*i*j*trace_y)/length
        weight = .5/length
        x, y = .5*(trace_x[a]+trace_x[b]), .5*(trace_y[a]+trace_y[b])
        coherence = np.hypot(x, y)
        aligned = np.divide(.5*(direction[a]+direction[b]), coherence,
                            out=np.zeros_like(coherence), where=coherence > 1e-8)
        angular = ANGULAR_SCALE*np.power(.5*(1.+aligned.clip(-1., 1.)), 4)
        links.append((a, b, weight*(1.+(1.-cross)*coherence*(angular-1.))))
    return -np.expm1(-smooth_ink/.10), tuple(links)


@dataclass
class Paper:
    water: np.ndarray
    movable: np.ndarray
    retained: np.ndarray
    contact_water: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def initial(cls, field, moisture):
        water = field*moisture
        active = field*(.40*water/(1.+water))
        return cls(water, active, field-active)

    def ink(self):
        return self.retained+self.movable

    def advance(self, ink, elapsed, *, width_px, spread, moisture, uniform=False, cross=TRACE_CROSS, added_water=None,
                spacing=(1., 1.)):
        """Receive positive optical dose and move pigment only while it remains mobile."""
        if (ink.shape != self.water.shape or (ink < 0.).any()
                or not np.isfinite(ink).all() or not np.isfinite((elapsed, width_px, spread, moisture)).all()
                or elapsed <= 0. or width_px <= 0. or min(spread, moisture) < 0.
                or (cross is not None and not 0. <= cross <= 1.)):
            raise ValueError("Expected a matching nonnegative ink source and finite paper transport")
        diffusion = spread*width_px**2
        water_diffusion = diffusion*WATER_TO_PIGMENT_SPREAD
        water_source = moisture*ink if added_water is None else np.asarray(added_water, dtype=float)
        if water_source.shape != self.water.shape or not np.isfinite(water_source).all() or (water_source < 0.).any():
            raise ValueError("Water delivery must be finite, nonnegative and on the same paper grid")
        dx, dy = spacing
        if not np.isfinite((dx, dy)).all() or min(dx, dy) <= 0.:
            raise ValueError("Paper cell spacing must be finite and positive")
        guided = not uniform and cross is not None and water_diffusion > 0. and min(ink.shape) >= 2
        # The sum of incident directional links is bounded by 3*ANGULAR_SCALE.
        # A smaller explicit step preserves nonnegative water and pigment.
        limit = (.90/(3.*ANGULAR_SCALE) if guided else .22)*min(dx, dy)**2
        n = max(1, int(np.ceil(elapsed*water_diffusion/limit)))
        dt = elapsed/n
        input_ = ink/n
        water_in = water_source/n
        links = tuple((a, b, 1./pitch**2) for (a, b), pitch in zip(GRID, (dx, dy), strict=True))
        for step in range(n):
            self.water += water_in
            self.movable += input_
            if guided:
                if step == 0:
                    trace, geometry = trace_guide(self.ink(), width_px, cross, spacing)
                # Water can open a link into a narrow blank. Wider dry gaps
                # begin with only the low permeability of fresh paper.
                aperture = 1.-(1.-trace)*(.20/(.20+self.water))
                links = tuple((a, b, w*(DRY_LINK+(1.-DRY_LINK)*np.minimum(aperture[a], aperture[b])))
                              for a, b, w in geometry)
            exchange(self.water, dt*water_diffusion, links=links)
            wet = self.water/(1.+self.water)
            mobility = None if uniform else wet
            exchange(self.movable, dt*diffusion, mobility, links=links)
            fixed_fraction = -np.expm1(-dt*BINDING*(1.-wet))
            fixed = self.movable*fixed_fraction
            self.retained += fixed
            self.movable -= fixed
            if self.contact_water is not None:
                mobile_water, bound_water = self.contact_water
                mobile_water += input_*wet
                exchange(mobile_water, dt*diffusion, mobility, links=links)
                captured = mobile_water*fixed_fraction
                bound_water += captured
                mobile_water -= captured
            self.water *= np.exp(-dt*ABSORPTION)


def response(initial, increments, *, shape_width, spread, moisture, uniform=False, cross=TRACE_CROSS):
    """Return the final local paper state and a trace of the wet area."""
    paper = Paper.initial(initial, moisture)
    wet_area = []
    for dt, ink in increments:
        paper.advance(ink, dt, width_px=shape_width, spread=spread, moisture=moisture, uniform=uniform, cross=cross)
        wet_area.append(int((paper.water >= .20).sum()))
    return paper, wet_area


def fit(cases):
    """Learn nonnegative mixing weights for conservative paper responses."""
    regressors, target = [], []
    for fields, truth, former_white, interior in cases:
        scale = max(.05, np.mean(truth))
        normalize = np.sqrt(truth.size)*scale
        design = fields.reshape(len(fields), -1).T
        data = [design/normalize]
        values = [truth.ravel()/normalize]
        for zone in (former_white & interior, former_white & ~interior):
            if zone.any():
                count = max(1., np.sqrt(zone.sum()))
                norm = count*scale
                data.append(fields[:, zone].sum(axis=1)[None]/norm)
                values.append(np.array([truth[zone].sum()/norm]))
        regressors.append(np.concatenate(data))
        target.append(np.concatenate(values))
    x, y = np.concatenate(regressors), np.concatenate(target)

    def loss(w):
        delta = x@w-y
        return float(.5*delta@delta), x.T@delta

    start = np.r_[1., np.zeros(len(SPREAD)-1)]
    constraints = {"type": "eq", "fun": lambda w: w.sum()-1., "jac": lambda w: np.ones_like(w)}
    fitted = minimize(loss, start, method="SLSQP", jac=True, bounds=[(0., 1.)]*len(start),
                      constraints=constraints, options={"ftol": 1e-10, "maxiter": 100})
    if not fitted.success or not np.isclose(fitted.x.sum(), 1.):
        raise ValueError(f"Local wetting fit did not converge: {fitted.message}")
    weights = fitted.x.clip(0.)
    return dict(weights=weights/weights.sum(), loss=np.array((loss(start)[0], fitted.fun)), spread=SPREAD.copy())
