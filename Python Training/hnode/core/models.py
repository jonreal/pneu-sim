# hnode/core/models.py
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class NetForceModel(eqx.Module):
    F_net: eqx.nn.MLP

    def __init__(self, *, key, width: int, depth: int):
        self.F_net = eqx.nn.MLP(4, 1, width, depth, activation=jax.nn.leaky_relu, key=key)

    def __call__(self, mf, me, displacement, velocity):
        net_input = jnp.stack([mf, me, displacement, velocity])
        F = self.F_net(net_input).squeeze()
        return F

class HybridSystem(eqx.Module):
    force_net: NetForceModel
    params: jnp.ndarray  # [mass, damping, C, nu]
    r0    : float
    L0    : float

    def __init__(self, *, key, width=96, depth=2,
                 init_params=(253.0, 84700000.0, 4.6),
                 r0: float = 5.0, L0: float = 200.0):
        key1, _ = jr.split(key)
        self.force_net = NetForceModel(key=key1, width=width, depth=depth)
        self.params = jnp.array(init_params)
        self.r0, self.L0 = float(r0), float(L0)

    def __call__(self, t, y, u_fn, mf, me):
        m  = jax.nn.softplus(self.params[0])
        b  = 0
        C  = jax.nn.softplus(self.params[1])
        nu = jax.nn.softplus(self.params[2])
        
        x, dx, Pf, Pe = y
        u = u_fn(t)

        F_nn  = self.force_net(mf, me, x, dx)

        x_dot  = dx
        dx_dot = (u - b * dx - F_nn) / m

        rf = lambda x_: self.r0 + nu*(self.r0/self.L0)*x_
        re = lambda x_: self.r0 - nu*(self.r0/self.L0)*x_

        Vf  = lambda x_: jnp.pi * rf(x_)**2 * (self.L0 - x_)
        dVf = lambda x_, dx_:  jnp.pi*dx_ * (2*rf(x_)*nu*(self.r0/self.L0)*(self.L0 - x_) - rf(x_)**2)
        Ve  = lambda x_: jnp.pi * re(x_)**2 * (self.L0 + x_)
        dVe = lambda x_, dx_: -jnp.pi*dx_ * (2*re(x_)*nu*(self.r0/self.L0)*(self.L0 + x_) - re(x_)**2)

        Pf_dot = - C * mf * dVf(x,dx) / Vf(x)**2
        Pe_dot = - C * me * dVe(x,dx) / Ve(x)**2

        return jnp.array([x_dot, dx_dot, Pf_dot, Pe_dot])
