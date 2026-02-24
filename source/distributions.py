import torch
import math

R = 1.0    # main radius
r = 0.2    # tube radius

def make_init_dist(n_samples: int = 1000, sigma: float = 0.05):
    """
    Initial: single Gaussian on torus at theta=0, phi=0 (x=1, y=0, z=0)
    Returns (theta, phi)
    """
    theta = torch.randn(n_samples) * sigma + 0.0
    phi = torch.randn(n_samples) * sigma + 0.0
    theta %= 2*math.pi
    phi %= 2*math.pi
    return torch.stack([theta, phi], dim=1)


def make_final_dist(n_samples: int = 1000, sigma: float = 0.05):
    n_half = n_samples // 2

    # Top Gaussian
    theta_top = torch.randn(n_half) * sigma + 3*math.pi/4
    phi_top = torch.randn(n_half) * sigma + math.pi/2

    # Bottom Gaussian
    theta_bottom = torch.randn(n_half) * sigma + 5*math.pi/4
    phi_bottom = torch.randn(n_half) * sigma - math.pi/2
    """
    theta_bottom2 = torch.randn(n_half) * sigma + math.pi/2
    phi_bottom2 = torch.randn(n_half) * sigma + math.pi/2

    theta_bottom3 = torch.randn(n_half) * sigma
    phi_bottom3 = torch.randn(n_half) * sigma + math.pi
    """
    # Wrap angles
    theta_top %= 2*math.pi
    phi_top %= 2*math.pi
    theta_bottom %= 2*math.pi
    phi_bottom %= 2*math.pi
    """
    theta_bottom2 %= 2*math.pi
    phi_bottom2 %= 2*math.pi
    theta_bottom3 %= 2*math.pi
    phi_bottom3 %= 2*math.pi
    """
    theta = torch.cat([theta_top, theta_bottom])#, theta_bottom2, theta_bottom3])
    phi = torch.cat([phi_top, phi_bottom])# phi_bottom2, phi_bottom3])

    return torch.stack([theta, phi], dim=1)
