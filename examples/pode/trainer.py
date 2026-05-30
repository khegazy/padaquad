from __future__ import annotations

import hashlib
import json
import os
import time
from glob import glob

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchdiffeq import odeint

import padaquad as tpd

from curriculum import CurriculumClass


def _make_serializable(obj):
    """Convert a config dict into a JSON-serializable form for hashing and saving."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, torch.nn.Module):
        return type(obj).__name__
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return float(obj) if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class Trainer(CurriculumClass):
    def __init__(
        self,
        model,
        integrator_config,
        t_max,
        loss_fxn="MSE",
        lr=1e-3,
        N_epochs=None,
        t_pred=0.1,
        t_init=0.0,
        curr_type=None,
        curr_config=None,
        lr_patience=None,
        lr_scale=None,
        dtype=torch.float64,
        device="cuda",
    ) -> None:
        super().__init__(
            curr_type=curr_type,
            t_pred=t_pred,
            t_max=t_max,
            config=curr_config,
            dtype=dtype,
        )
        self.model = model
        self.loss_fxn_name = loss_fxn
        self.plot_colors = ["k", "b", "r", "g"]
        self.device = device

        assert N_epochs is not None or t_max is not None
        self.N_epochs = np.inf if N_epochs is None else N_epochs

        lr_check = lr_patience is not None and lr_scale is not None
        lr_check = lr_check or (lr_patience is None and lr_scale is None)
        assert lr_check, "Must specify both 'lr_scal' and 'lr_patience'"
        self.lr_patience = lr_patience
        self.lr_scale = lr_scale

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.init_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), tolerance_change=0.0, line_search_fn="strong_wolfe"
        )

        assert hasattr(self, f"_{loss_fxn}")
        self.loss_fxn = getattr(self, f"_{loss_fxn}")

        self.integrator = tpd.UniformAdaptiveQuadrature(
            **integrator_config, device=self.device
        )

    def _find_saved_weights(self):
        files = glob(os.path.join(self.chkpt_dir, "*pth"))
        if len(files) == 0:
            return None, None
        max_t = -1.0
        filename = None
        for f in files:
            idx = f.find("_t-") + 3
            print(f, f[idx:-4])
            t_str = f[idx:-4]
            t = float(t_str)
            if t > max_t:
                max_t = t
                filename = f
                t_label = t_str
        return filename, t_label

    @staticmethod
    def _is_integer_t(filepath):
        """Check if a checkpoint/results file has an integer t value."""
        idx = filepath.find("_t-") + 3
        t_str = filepath[idx:].split(".h5")[0].split(".pth")[0]
        t_val = float(t_str)
        return t_val == int(t_val)

    def _cleanup_non_integer_files(self, directory, pattern):
        """Remove files with non-integer t values, keeping integer-t milestones."""
        current_file = os.path.basename(
            self._get_chkpt_filename()
            if pattern == "*pth"
            else self._get_results_filename()
        )
        for f in glob(os.path.join(directory, pattern)):
            if os.path.basename(f) == current_file:
                continue
            if not self._is_integer_t(f):
                os.remove(f)

    def _save_model(self, t, epoch_count=0):
        filename = os.path.join(self.chkpt_dir, self._get_chkpt_filename())
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch_count,
                "t_pred": self.t_pred,
                "prev_t_pred": self.prev_t_pred,
            },
            filename,
        )
        # Clean up non-integer-t checkpoints and results
        self._cleanup_non_integer_files(self.chkpt_dir, "*pth")
        self._cleanup_non_integer_files(self.results_dir, "*.h5")

    def _get_results_filename(self):
        t_label = float(np.squeeze(self.t_pred))
        return f"training_t-{t_label:.3f}.h5"

    def _get_chkpt_filename(self):
        t_label = float(np.squeeze(self.t_pred))
        return f"model_weights_t-{t_label:.3f}.pth"

    def _load_model(self):
        filename, t_label = self._find_saved_weights()
        print("FILENAME", filename)
        if filename is None:
            print("Start training from scratch")
            return 0
        else:
            print(f"Starting training from {filename}")
            checkpoint = torch.load(filename)
            # Support both old format (bare state_dict) and new format (dict with keys)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # Restore curriculum state
                if "t_pred" in checkpoint:
                    self.t_pred = checkpoint["t_pred"]
                if "prev_t_pred" in checkpoint:
                    self.prev_t_pred = checkpoint["prev_t_pred"]
                epoch = checkpoint.get("epoch", 0)
            else:
                self.model.load_state_dict(checkpoint)
                epoch = 0

            # Restore eval history from results file if available
            results_filename = self._get_results_filename()
            results_path = os.path.join(self.results_dir, results_filename)
            if os.path.exists(results_path):
                with h5.File(results_path, "r") as f:
                    self.eval_integral_values = list(f["integrals"])
                    self.eval_integral_limits = list(f["t_final"])
                    if epoch == 0:
                        epoch = int(f["epoch"][...])
                    if "t_pred" not in checkpoint:
                        self.t_pred = f["t_pred"][...]
                        idx = -1
                        while self.eval_integral_limits[idx][0] >= self.t_pred.item():
                            idx -= 1
                        self.prev_t_pred = self.eval_integral_limits[idx][0]

            print(f"Resuming from epoch {epoch}, t_pred={self.t_pred}")
            return epoch

    def plot_results(self, time, pred, solution, epoch):
        device = pred.device
        pred = pred.cpu()
        solution = solution.cpu()
        time = time.cpu()
        plt.figure(figsize=(10, 7))
        colors = ["red", "green", "purple", "orange"]
        plt.plot(
            time[:, 0].detach().numpy(),
            pred[:, 0].detach().numpy(),
            label="Predicted x",
            color=colors[0],
            lw=2.5,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            pred[:, 1].detach().numpy(),
            label="Predicted y",
            color=colors[1],
            lw=2.5,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            solution[:, 0].detach().numpy(),
            label="Ground Truth x",
            color=colors[2],
            linestyle="--",
            lw=2,
            alpha=0.8,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            solution[:, 1].detach().numpy(),
            label="Ground Truth y",
            color=colors[3],
            linestyle="--",
            lw=2,
            alpha=0.8,
        )
        # plt.title('Lotka-Volterra learned ODE', fontsize=16)
        (
            plt.ylabel("Prey/Predator Population (x/y)", fontsize=16),
            plt.xlabel("Time (t)", fontsize=16),
        )
        plt.legend(fontsize=13), plt.grid(True)  # plt.axis('equal')
        plt.xticks(fontsize=13)  # Sets x-axis tick label font size
        plt.yticks(fontsize=13)  # Sets y-axis tick label font size
        plt.xlim(0, 20)
        plt.ylim(0, 3)

        plt.savefig(os.path.join(self.plot_dir, f"ppr_training_{epoch}.png"))
        pred.to(device)
        solution.to(device)

    def _plot_results(self, time, pred, solution, epoch):
        N_vars = pred.shape[-1]
        fig, axes = plt.subplots(2, 1)
        y_max, y_min = -np.inf, np.inf
        for i in range(N_vars):
            if torch.amax(solution[:, i]) > y_max:
                y_max = torch.amax(solution[:, i])
            if torch.amax(solution[:, i]) < y_min:
                y_min = torch.amin(solution[:, i])
            for j in range(2):
                axes[j].plot(
                    time[:, 0], solution[:, i], color=self.plot_colors[i], alpha=0.5
                )
                axes[j].plot(
                    time[:, 0],
                    pred[:, i].detach().numpy(),
                    color=self.plot_colors[i],
                    linestyle=":",
                )
        y_delta = 0.05 * np.abs(y_max - y_min)
        if y_delta > 0.1 * np.abs(y_max):
            y_max += y_delta
        else:
            y_max += 0.1 * np.abs(y_max)

        if y_delta > 0.1 * np.abs(y_min):
            y_min -= y_delta
        else:
            y_min += 0.1 * np.abs(y_min)

        for i in range(2):
            axes[i].set_xlim(time[0], time[-1])

        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(np.maximum(y_min, 1e-9), y_max)
        axes[1].set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, f"training_{epoch}.png"))

    def eval_results(self, t_init, epoch_count):
        t_eval_max = self.t_pred if self.t_max == np.inf else self.t_max
        t_eval = torch.linspace(
            t_init, t_eval_max, 1000, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        eval_output = odeint(
            self.ode_fxn.ode, self.ode_fxn.initial_condition[0], t_eval[:, 0]
        )
        self.plot_results(t_eval, self.model(t_eval), eval_output, epoch_count)

    @staticmethod
    def _MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        return torch.sum((pred - target) ** 2, dim=1, keepdim=True)

    @staticmethod
    def _MAE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        return torch.sum(torch.abs(pred - target), dim=1, keepdim=True)

    @staticmethod
    def _relative_MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        # print("diff", (pred - target))
        # print("pred", pred)
        # print("targ", target)
        return torch.sum(
            ((pred - target) / (torch.abs(target) + 1e-2))
            ** 2,  # TODO: make eps smaller
            dim=1,
            keepdim=True,
        )

    def _MSE_combined(self, pred, target):
        return self._MSE(pred, target) + self._relative_MSE(pred, target)

    def _MSE_MAE(self, pred, target):
        return self._MSE(pred, target) + self._MAE(pred, target)

    def solution_integrad(self, t, model, verbose=False):
        pred = model(t)

        # TODO: use gaussian instead and t_init
        # scale = torch.exp(-1*t)
        # scale /= 1.0 - torch.exp(-1*torch.tensor(self.t_pred))
        if verbose:
            print("pred", pred)
            print("targ", self.ode_fxn(t, pred))
            # print("scale", scale)
        return self.loss_fxn(pred, self.ode_fxn(t, pred))

    def _integrad(self, t, model, verbose=False):
        pred, target = self.ode_fxn.train_eval(model, t)

        # TODO: use gaussian instead and t_init
        # scale = torch.exp(-1*t)
        # scale /= 1.0 - torch.exp(-1*torch.tensor(self.t_pred))
        """
        if verbose:
            print("pred", jac)
            print("targ", self.ode_fxn(t, model(t)))
            #print("scale", scale)
        """
        return self.loss_fxn(pred, target)

    def train_iteration(self):
        self.optimizer.zero_grad()
        self.integral_output = self.integrator.integrate(
            f=self._integrad, mesh=self.input_times, f_args=(self.model,)
        )
        # print(epoch_count, integral_output.nodes.shape)
        loss = self.integral_output.loss
        # print("BEFORE", self.model.layers[0].weight.data[:5])
        # print("OUTPUT", integral_output)

        # T0 loss
        init_loss = torch.mean(
            self.loss_fxn(self.model(self.t_init_eval), self.ode_fxn.initial_condition)
        )
        loss = loss + init_loss
        if not self.integral_output.gradient_taken:
            # print("taking gradients")
            loss.backward()
        else:
            init_loss.backward()
        return loss

    def train(self, config, ode_fxn, experiment_name=None):
        self.ode_fxn = ode_fxn
        self.ode_fxn.solve_ode(self.t_max)
        t_init = self.ode_fxn.t_init
        t_init_eval = torch.tensor(
            [t_init], requires_grad=True, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.prev_t_pred = 1e-9

        serializable_config = _make_serializable(config)
        config_vars = json.dumps(serializable_config, sort_keys=True).encode()
        hash_label = hashlib.blake2s(config_vars, digest_size=4).hexdigest()

        # Directory structure: <problem>/<loss_fxn>/<experiment_name>_<hash>/
        # e.g. lotka_volterra/MSE/lotka_volterra_a1b2c3d4/
        problem_name = self.ode_fxn.__name__
        if experiment_name is not None:
            run_dir = f"{experiment_name}_{hash_label}"
        else:
            run_dir = hash_label
        sub_dir = os.path.join(problem_name, self.loss_fxn_name, run_dir)
        self.chkpt_dir = os.path.join("checkpoints", sub_dir)
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.results_dir = os.path.join("results", sub_dir)
        self.plot_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.eval_integral_values, self.eval_integral_limits = [], []

        print(f"Checkpoints: {self.chkpt_dir}")
        print(f"Results:     {self.results_dir}")

        for d in (self.results_dir, self.chkpt_dir):
            with open(os.path.join(d, "config.yaml"), "w") as f:
                yaml.dump(serializable_config, f, default_flow_style=False)
        epoch_count = self._load_model()

        self.model.train()
        # self.model.compile()

        print("Optimizer Parameters:")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"\nParameter Group {i}:")
            for key, value in param_group.items():
                if key == "params":
                    # 'params' contains a list of actual parameter tensors
                    print(f"  {key}: {len(value)} tensors")
                    for j, param in enumerate(value):
                        print(
                            f"    Tensor {j} - Shape: {param.shape}, Requires Grad: {param.requires_grad}"
                        )
                        # You can also print the tensor values if desired (be mindful of large tensors)
                        # print(f"      Value: {param.data}")
                else:
                    print(f"  {key}: {value}")
        self.t_init_eval = t_init_eval

        # Train t0 (skip if resuming from checkpoint)
        if epoch_count == 0:
            loss_ratio, error_ratio, t0_count = 1, torch.tensor([1e10]), 0
            prev_loss, prev_filename = 1.0, None
            pred = torch.ones(1)
            print("Training initial conditions")
            while torch.any(error_ratio > 1e-3):
                if t0_count % 100 == 0:
                    print(
                        f"T0 count {t0_count} | loss ratio: {loss_ratio} / error ratio: {error_ratio} / prediction: {torch.squeeze(pred).item()}"
                    )
                self.init_optimizer.zero_grad()
                pred = self.model(t_init_eval)
                loss = torch.mean(self.loss_fxn(pred, self.ode_fxn.initial_condition))
                # print("LOSS 0", loss.shape, loss)
                # print(self.model(t_init), self.ode_fxn(t_init))
                loss.backward()
                # print("BEFORE", self.model.layers[0].bias.data[:5])
                # print("GRAD", self.model.layers[0].bias.grad[:5]*1e5)
                self.init_optimizer.step()
                loss_ratio = torch.abs(prev_loss - loss) / prev_loss
                error_ratio = torch.abs(pred - self.ode_fxn.initial_condition) / (
                    torch.abs(self.ode_fxn.initial_condition) + 1e-9
                )
                error_ratio = torch.squeeze(error_ratio)
                print(
                    "WTF",
                    self.ode_fxn.initial_condition.shape,
                    pred.shape,
                    error_ratio.shape,
                )
                prev_loss = loss
                # print("AFTER", self.model.layers[0].bias.data[:5])
                # print(t0_count, self.t_pred, loss, loss_ratio)
                t0_count += 1
            print("Finished training initial conditions", loss_ratio, error_ratio)
        else:
            print(f"Skipping IC training (resuming from epoch {epoch_count})")

        # Make output folder
        prev_loss = 0
        N_const_loss = 0
        cutoff_count = 0
        self.input_times = t_init_eval
        train_criteria = True
        self.integral_output = None
        force_update = False
        last_save_t_pred = float(np.squeeze(self.t_pred))
        while train_criteria:
            updated_curr = self.integral_output is not None and self.update_curriculum(epoch_count, loss, force_update)
            curr_t_pred = float(np.squeeze(self.t_pred))
            time_to_eval = (curr_t_pred - last_save_t_pred) >= 0.1
            time_to_save = time_to_eval and (updated_curr and not force_update)
            force_update = False

            if epoch_count % 10 == 0 and self.integral_output is not None:
                print(
                    f"Epoch/Time {epoch_count}/{self.t_pred}: {loss.item()} | {self.config['cutoff']} | {len(self.integral_output.nodes)}"
                )
            if time_to_eval:
                # print("INIT", t_init, torch.squeeze(self.model(t_init_eval)).detach().numpy())
                if self.integral_output is not None:
                    # self.loss_integrad(torch.arange(5, dtype=self.dtype).unsqueeze(-1)*5./4., self.model, verbose=True)
                    # if integral_output is not None:
                    #    print(integral_output.nodes[:,0,0])
                    integral_values = [loss.item()]
                    integral_limits = [self.t_pred]
                    eval_t_preds = np.concatenate(
                        [
                            np.array([self.prev_t_pred]),
                            self.t_pred * np.array([0.95, 0.9, 0.75, 0.5, 0.25]),
                        ]
                    )
                    # print("EVAL", self.t_pred, eval)
                    half_idx = -2
                    for eval_t in eval_t_preds:
                        self.eval_input_times = torch.tensor(
                            [t_init, eval_t], dtype=self.dtype, device=self.device
                        ).unsqueeze(-1)
                        with torch.no_grad():
                            eval_integral_output = self.integrator.integrate(
                                f=self._integrad,
                                mesh=self.eval_input_times,
                                f_args=(self.model,),
                            )
                        integral_values.append(eval_integral_output.loss.item())
                        integral_limits.append(eval_t)
                    self.eval_integral_values.append(np.array(integral_values))
                    data_values = np.array(self.eval_integral_values)
                    self.eval_integral_limits.append(np.array(integral_limits))
                    data_limits = np.array(self.eval_integral_limits)
                    filename = os.path.join(
                        self.results_dir, self._get_results_filename()
                    )
                    with h5.File(filename, "w") as f:
                        f.create_dataset("epoch", data=epoch_count)
                        f.create_dataset("t_pred", data=self.t_pred)
                        f.create_dataset("t_final", data=data_limits)
                        f.create_dataset("integrals", data=data_values)
                    # if prev_filename is not None:
                    #     print("preve filename", prev_filename)
                    #     os.remove(prev_filename)
                    # prev_filename = filename

                    fig, ax = plt.subplots(2, 1, height_ratios=[5, 1])
                    ax[0].plot(data_values[:, -2], "y--")
                    ax[0].plot(data_values[:, -3], "g--")
                    ax[0].plot(data_values[:, -4], "r--")
                    ax[0].plot(data_values[:, 1], "b--")
                    ax[0].plot(data_values[:, 0], "k-")
                    ax[1].plot(data_limits[:, 0])
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.results_dir, "training_progress.png"))
                    plt.close(fig)

                    if integral_values[half_idx] / integral_values[0] > 0.25:
                        cutoff_count = cutoff_count + 1
                        if cutoff_count > 50:
                            self.config["cutoff"] = 1.25 * self.config["cutoff"]
                            cutoff_count = 0
                    else:
                        cutoff_count = 0

                    self._integrad(
                        torch.arange(
                            10, dtype=self.dtype, device=self.device
                        ).unsqueeze(-1)
                        * self.t_pred
                        / 9.0,
                        self.model,
                        verbose=True,
                    )
                    #if self.integral_output is not None:
                    #    print("Y", self.integral_output.y[0, :, 0])
                    self.eval_results(t_init, epoch_count)
                    if time_to_save:
                        self._save_model(self.t_pred, epoch_count)

                last_save_t_pred = np.squeeze(self.t_pred)

            if updated_curr:
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    tolerance_change=0.0,
                    line_search_fn="strong_wolfe",
                )

            if updated_curr or self.integral_output is None:
                if self.input_times[-1] < self.t_pred:
                    self.input_times = torch.concatenate(
                        [
                            self.input_times,
                            torch.tensor([self.t_pred], device=self.device).unsqueeze(
                                -1
                            ),
                        ],
                        dim=0,
                    )
                self.input_times = torch.tensor(
                    [t_init, self.t_pred], dtype=self.dtype, device=self.device
                ).unsqueeze(-1)
            else:
                self.input_times = self.integral_output.mesh_optimal
            # self.input_times = torch.tensor([t_init, self.t_pred], dtype=self.dtype, device=self.device).unsqueeze(-1)
            # print("TIMES", times.shape, loss, torch.std(self.loss_history), self.loss_history, times)
            """
            integral_output = self.integrator.integrate(
                f=self._integrad, mesh=self.input_times, f_args=(self.model,)
            )
            #print(epoch_count, integral_output.nodes.shape)
            loss = integral_output.loss
            #print("BEFORE", self.model.layers[0].weight.data[:5])
            #print("OUTPUT", integral_output)
            if not integral_output.gradient_taken:
                #print("taking gradients")
                loss.backward()
            """
            """

            t_eval = torch.arange(100).unsqueeze(1)*self.t_pred/99
            loss = torch.mean(
                self.loss_fxn(self.model(t_eval), self.ode_fxn(t_eval))
            )
            loss.backward()
            """

            # print("GRAD", self.model.layers[0].weight.grad[:5]*1e5)
            loss = self.optimizer.step(self.train_iteration)
            if loss == prev_loss:
                N_const_loss += 1
            else:
                prev_loss = loss
                N_const_loss = 0
            if N_const_loss > 3:
                print("FORCING UPDATE")
                force_update = True
            if N_const_loss > 0:
                #print("NEW SOLVER")
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    tolerance_change=0.0,
                    line_search_fn="strong_wolfe",
                )
                # asdfasd
            # print("AFTER", self.model.layers[0].weight.data[:5])
            # times = integral_output.mesh_optimal.clone()
            # times = integral_output.mesh_optimal.detach()
            # times.requires_grad_(True)

            # # Update learning rate
            # if self.t_pred != self.t_max and self.lr_patience is not None:
            #     if loss < prev_loss or updated_curr:
            #         patience = 0
            #     else:
            #         patience += 1
            #         if patience >= self.lr_patience:
            #             self.optimizer.lr = self.optimizer.lr*self.lr_scale
            #             patience = 0
            #     prev_loss = loss.detach().item()

            epoch_count += 1
            if self.N_epochs is not None:
                train_criteria = epoch_count < self.N_epochs
            else:
                train_criteria = self.input_times[-1, 0] < self.t_max

        self.eval_results(t_init, epoch_count)
