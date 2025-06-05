import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import time


# Hybrid Neural ODE system: partial known dynamics + learned force
class HybridSystem(eqx.Module):
    force_net: eqx.nn.MLP
    params: jnp.ndarray  # [p1, p2]

    def __init__(self, *, key, width=64, depth=2):
        key1, key2 = jr.split(key)
        self.force_net = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=width, depth=depth,
            activation=jax.nn.relu, key=key1
        )
        self.params = jnp.array([0.5, 0.5])

    def __call__(self, t, y, args):
        u_fn = args  # u_fn(t)
        p = jax.nn.softplus(self.params)  # enforce positivity
        u = u_fn(t)
        f_partial = jnp.array([
            y[1],
            1/p[0] * (u - p[1]*y[1])
        ])
        f_learned = jnp.array([0.0, self.force_net(y).squeeze()])
        return f_partial + f_learned


def generate_training_data(n_samples, ts, key):
    p = [0.5, 0.5, -0.03, 0.1, 0.5]
    x0 = jnp.array([0.0, 0.0])
    data = []
    keys = jr.split(key, n_samples)

    for i in range(n_samples):
        amp = jr.uniform(keys[i], ())
        u_fn = lambda t, a=amp: a * (t >= 1.0)

        def f(t, y, _):
            u = u_fn(t)
            return jnp.array([
                y[1],
                1/p[0] * (u - p[1]*y[1] - (p[2]*y[1] + p[3]*y[0] + p[4]*y[0]**2))
            ])

        term = diffrax.ODETerm(f)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(
            term, solver, ts[0], ts[-1], dt0=0.1,
            y0=x0, saveat=diffrax.SaveAt(ts=ts), args=u_fn,
            adjoint=diffrax.BacksolveAdjoint())
        data.append((x0, amp, sol.ys))

    return data


def main():
    T = 10.0
    dt = 0.1
    ts = jnp.arange(0.0, T, dt)
    n_samples = 128
    n_epochs = 10
    lr = 1e-2
    key = jr.PRNGKey(0)

    #adjoint = diffrax.BacksolveAdjoint()
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    # Generate data
    data = generate_training_data(n_samples, ts, key)

    # Initialize model
    model_key, train_key = jr.split(key)
    model = HybridSystem(key=model_key)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(model, x0, amp, y_true):
        u_fn = lambda t: amp * (t >= 1.0)
        term = diffrax.ODETerm(model)
        sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), ts[0], ts[-1], dt0=0.1,
                                  y0=x0, saveat=diffrax.SaveAt(ts=ts), args=u_fn)
        return jnp.mean((sol.ys - y_true)**2)

    @eqx.filter_jit
    def train_step(model, opt_state, x0, amp, y_true):
        loss, grads = loss_fn(model, x0, amp, y_true)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Plot ground truth
    fig, ax = plt.subplots()
    lines = [ax.plot(ts, y[:, 0], '-', label=f'Target {i}')[0] for i, (_, _, y) in enumerate(data)]
    preds = [ax.plot(ts, ts * 0, '--', label=f'Pred {i}')[0] for i in range(n_samples)]
    # ax.legend()  # Legend disabled

    # Training loop
    print("    Iteration    Epoch    TimeElapsed    LearnRate    TrainingLoss")
    print("    _________    _____    ___________    _________    ____________")
    for epoch in range(n_epochs):
        start_epoch_time = time.time()
        total_loss = 0.0
        for x0, amp, y_true in data:
            model, opt_state, loss = train_step(model, opt_state, x0, amp, y_true)
            total_loss += loss

        elapsed = time.time() - start_epoch_time
        lr = 1e-3  # fixed for now; adjust if dynamic
        print(f"    {epoch+1:9d}    {epoch+1:5d}       {elapsed:08.2f}   {lr:.8f}       {(total_loss / n_samples):.7f}")

        for i, (x0, amp, _) in enumerate(data):
            u_fn = lambda t, a=amp: a * (t >= 1.0)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(model), diffrax.Tsit5(), ts[0], ts[-1], dt0=0.1,
                y0=x0, saveat=diffrax.SaveAt(ts=ts), args=u_fn
            )
            preds[i].set_ydata(sol.ys[:, 0])

        plt.draw()
        plt.pause(0.1)

    # Plot ForceNet force as a surface
    from mpl_toolkits.mplot3d import Axes3D
    fig_force = plt.figure()
    ax_force = fig_force.add_subplot(111, projection='3d')

    all_y = jnp.concatenate([y for (_, _, y) in data], axis=0)
    y0_min, y0_max = all_y[:, 0].min(), all_y[:, 0].max()
    y1_min, y1_max = all_y[:, 1].min(), all_y[:, 1].max()

    margin = 0.1 * max(y0_max - y0_min, y1_max - y1_min)
    y0_vals = jnp.linspace(y0_min - margin, y0_max + margin, 50)
    y1_vals = jnp.linspace(y1_min - margin, y1_max + margin, 50)
    Y0, Y1 = jnp.meshgrid(y0_vals, y1_vals)
    Y_input = jnp.stack([Y0.ravel(), Y1.ravel()], axis=1)

    forces = jax.vmap(model.force_net)(Y_input).reshape(50, 50)

    ax_force.plot_surface(Y0, Y1, forces, cmap='viridis', alpha=0.6, label='Learned')

    # Add ground truth force
    def true_force(y):
        p = [0.5, 0.5, -0.03, 0.1, 0.5]
        return 1/p[0] * (-p[2]*y[1] - p[3]*y[0] - p[4]*y[0]**2)

    true_vals = jnp.array([true_force(y) for y in Y_input]).reshape(50, 50)
    ax_force.plot_surface(Y0, Y1, true_vals, cmap='coolwarm', alpha=0.6, label='True')
    ax_force.set_title("Learned ForceNet Output")
    ax_force.set_xlabel("y[0]")
    ax_force.set_ylabel("y[1]")
    ax_force.set_zlabel("force")

    plt.show()


if __name__ == "__main__":
    main()
