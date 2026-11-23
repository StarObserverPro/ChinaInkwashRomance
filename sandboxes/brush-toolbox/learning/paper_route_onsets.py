"""Fit the existence, number and location of exterior white openings."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from linked_apertures import poisson_fit
from paper_routes import contexts


RELATIVE = np.array((0, 1, 2, 4, 5, 6, 8, 9))


def stroke_context(contact):
    length = len(contact.width)/contact.settled_width
    phase = np.linspace(0., 1., len(contact.width))
    middle = (phase >= .08) & (phase <= .92)
    tone = np.log1p(contact.mass/contact.width)[middle]
    center = np.median(tone)
    a, b = np.quantile(tone-center, (.1, .9))
    shape = np.log(contact.width[middle]/contact.settled_width)
    return np.array((1., np.log(length), a, b, np.std(shape), np.quantile(shape, .1))), length


def train(groups, routes):
    records = []
    for course, rows in groups.items():
        for mark in rows:
            _, contact, pose, signed = mark.supervision()
            x = np.arange(len(contact.width))[:, None]
            attributes = contexts(contact, x, np.array((-1, 1))[None], pose, signed)[..., RELATIVE]
            intensity = np.zeros(attributes.shape[:2])
            for path in routes[mark.name]:
                column = int(path.portal[0])
                if signed:
                    intensity[column, (path.side+1)//2] += 1.
                else:
                    intensity[column] += .5
            global_context, length = stroke_context(contact)
            records.append((attributes.reshape(-1, attributes.shape[-1]), intensity.ravel(),
                            global_context, length, 1./(len(groups)*len(rows)), contact.settled_width))
    design, hits, stroke, lengths, weights, widths = zip(*records, strict=True)
    relative, l1 = poisson_fit(np.concatenate(design), np.concatenate(hits),
                               np.concatenate([np.full(len(a), 1./w) for a, w in zip(design, widths, strict=True)]),
                               np.concatenate([np.full(len(a), w) for a, w in zip(design, weights, strict=True)]))
    counts = np.array([v.sum() for v in hits])
    s = np.asarray(stroke)
    w = np.array(weights)
    mean = w @ s
    mean[0] = 0.
    scale = np.sqrt(w @ (s-mean)**2+.04)
    x = (s-mean)/scale
    reported = counts > 0
    ridge = np.r_[.005, np.full(x.shape[1]-1, .08)]

    def likelihood(beta):
        logit = x @ beta
        score = w @ (np.logaddexp(0., logit)-reported*logit) + .5*np.sum(ridge*beta**2)
        gradient = (w*(expit(logit)-reported)) @ x+ridge*beta
        return float(score), gradient

    origin = np.zeros(x.shape[1])
    gate = minimize(likelihood, origin, jac=True, method="L-BFGS-B")
    if not gate.success:
        raise ValueError(f"Opening occurrence fit failed: {gate.message}")
    trimmed = [(a, y, float(wt), n) for a, y, wt, n in zip(design, hits, w, counts, strict=True) if n > 0]
    normal = sum(wt*n for _, _, wt, n in trimmed)

    def conditional(beta):
        loss = .04*(beta @ beta)
        grad = .08*beta
        for a, y, wt, n in trimmed:
            logits = a @ beta
            log_partition = logsumexp(logits)
            weight = wt/max(1e-9, normal)
            loss += weight*(log_partition*n-y @ logits)
            grad += weight*((np.exp(logits-log_partition)*n-y) @ a)
        return float(loss), grad

    start = np.zeros(design[0].shape[1])
    where = minimize(conditional, start, jac=True, method="L-BFGS-B")
    if not where.success:
        raise ValueError(f"Opening location fit failed: {where.message}")
    l = np.asarray(lengths)
    residual_rate = np.sum(w[reported]*(counts[reported]-1.))/np.sum(w[reported]*l[reported])
    return dict(relative=relative, mean=mean, scale=scale, gate=gate.x, location=where.x,
                extra=float(residual_rate), largest=float(counts.max()-1.), sources=np.array(list(groups)),
                loss=np.array((l1, [likelihood(origin)[0], gate.fun], [conditional(start)[0], where.fun])))


def rates(fitted, contact, conditions, method):
    reduced = conditions[..., RELATIVE]
    if method == "relative":
        value = np.exp((reduced @ fitted["relative"]).clip(-16., 5.))/contact.settled_width
        return value/value.sum(), float(value.sum()), None
    if method != "occurrence":
        raise ValueError("Choose a relative rate or a separate occurrence fit")
    feature, length = stroke_context(contact)
    probability = float(expit(((feature-fitted["mean"])/fitted["scale"]) @ fitted["gate"]))
    expected = min(fitted["largest"], length*fitted["extra"])
    logits = reduced @ fitted["location"]
    return np.exp(logits-logsumexp(logits)), probability*(expected+1.), (probability, expected)


def sample(fitted, contact, conditions, random, method):
    location, expected, hurdle = rates(fitted, contact, conditions, method)
    if hurdle is None:
        count = random.poisson(expected)
    else:
        probability, tail = hurdle
        count = (1+random.poisson(tail)) if random.random() < probability else 0
    selected = random.choice(location.size, size=count, p=location.ravel())
    columns, side = np.unravel_index(selected, location.shape)
    return [(column, 2*value-1) for column, value in zip(columns, side, strict=True)]
