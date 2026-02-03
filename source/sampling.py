import torch
import math
def integrate(x0, v, t=1.0, steps=100):
    dt = t/steps
    xs = [x0]
    x = x0
    for k in range(steps):
        t = torch.full((x.shape[0], 1), k * dt)
        x = x + dt * v(x, t)
        xs.append(x)
    return torch.stack(xs)

def integrate_torus(x0, v, t_end=1.0, steps=100):
    dt = t_end / steps
    xs = [x0]
    x = x0
    for k in range(steps):
        t = torch.full((x.shape[0], 1), k*dt)
        x = (x + dt * v(x, t)) % (2*math.pi)  # wrap angles
        xs.append(x)
    return torch.stack(xs)  # shape: (steps+1, batch, 2)
