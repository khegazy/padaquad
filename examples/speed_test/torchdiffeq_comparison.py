from __future__ import annotations

import logging
import time

import torch
from torchdiffeq import odeint

import padaquad as tpdiffeq
from padaquad.examples import (
    damped_sine,
    exp,
    sine_squared,
    t_squared,
    wolf_schlegel,
)

# Set up logging to both screen and file
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("speed_test.log"),
    ],
)

############################
#####  Test Variables  #####
############################
n_runs = 100
device = "cuda"
method = "dopri5"
atol = 1e-9
rtol = 1e-7
mesh_init = torch.tensor([0], dtype=torch.float64, device=device)
mesh_final = torch.tensor([1], dtype=torch.float64, device=device)
y0 = torch.tensor([0], dtype=torch.float64, device=device)

########################
#####  Integrands  #####
########################

integrands = [wolf_schlegel, damped_sine, sine_squared, exp, t_squared]

###################################
#####  torchdiffeq Speed Test #####
###################################
# Compare against torchdiffeq.odeint directly. The serial wrapper that
# previously lived in this library was removed in Phase 3 of the
# quadrature alignment plan; calling torchdiffeq directly is a more
# honest comparison anyway.
logger.info("torchdiffeq")
t_eval = torch.stack([mesh_init, mesh_final]).reshape(-1)
tdiffeq_results = []
for f in integrands:
    total_time = 0
    for _ in range(n_runs):
        t0 = time.time()
        _ = odeint(
            f,
            y0=y0,
            mesh=t_eval,
            method=method,
            atol=atol,
            rtol=rtol,
        )
        total_time = total_time + (time.time() - t0)
    tdiffeq_results.append(total_time / n_runs)


#######################################
#####  padaquad Speed Test #####
#######################################

logger.info("padaquad api")
tpdiffeq_api_results = []
for f in integrands:
    total_time = 0
    for _ in range(n_runs):
        t0 = time.time()
        _ = tpdiffeq.integrate(
            f=f,
            method=method,
            sampling="uniform",
            atol=atol,
            rtol=rtol,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            y0=y0,
            device=device,
        )
        total_time = total_time + (time.time() - t0)
    tpdiffeq_api_results.append(total_time / n_runs)


logger.info("padaquad integrator")
tpdiffeq_int_results = []
for f in integrands:
    total_time = 0
    integrator = tpdiffeq.adaptive_quadrature(
        sampling_type="uniform",
        f=f,
        method=method,
        atol=atol,
        rtol=rtol,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        y0=y0,
        device=device,
    )
    for _ in range(n_runs):
        t0 = time.time()
        _ = integrator.integrate()
        total_time = total_time + (time.time() - t0)
    tpdiffeq_int_results.append(total_time / n_runs)

# Log Results
message = "Problem \\ Method|    torchdiffeq    |    padaquad API\t  ratio    |    padaquad Integrator\t  ratio\n"
for idx, fxn in enumerate(integrands):
    td_out = tdiffeq_results[idx]
    tpd_api_out = tpdiffeq_api_results[idx]
    tpd_int_out = tpdiffeq_int_results[idx]
    ratio_api = td_out / tpd_api_out
    ratio_int = td_out / tpd_int_out
    message += f"{fxn.__name__}\t|      {td_out:.5f}      |"
    message += f"         {tpd_api_out:.5f}\t\t {ratio_api:.5f}   |"
    message += f"               {tpd_int_out:.5f}\t\t {ratio_int:.5f}\n"
logger.info("\n%s", message)
