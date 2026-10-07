"""
sir_model.py
------------
Core mathematical implementation of the SIR (Susceptible-Infected-Recovered) model.
Uses scipy.integrate.odeint for numerical integration of the differential equations.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SIRParameters:
    """Validated parameters for the SIR model simulation."""
    N: int            # Total population
    I0: int           # Initial infected
    R0_initial: int   # Initial recovered (renamed to avoid clash with R0 reproduction number)
    beta: float       # Infection rate
    gamma: float      # Recovery rate
    days: int         # Simulation duration
    vaccination_pct: float = 0.0  # Vaccination percentage (0–100)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate all parameters and raise descriptive errors on failure."""
        if self.N <= 0:
            raise ValueError("Total population N must be a positive integer.")
        if self.I0 < 0 or self.R0_initial < 0:
            raise ValueError("Initial infected and recovered must be non-negative.")
        if self.I0 + self.R0_initial > self.N:
            raise ValueError(
                f"I0 ({self.I0}) + R0 ({self.R0_initial}) must not exceed N ({self.N})."
            )
        if self.beta <= 0:
            raise ValueError("Infection rate (beta) must be greater than 0.")
        if self.gamma <= 0:
            raise ValueError("Recovery rate (gamma) must be greater than 0.")
        if self.days <= 0:
            raise ValueError("Simulation days must be a positive integer.")
        if not (0.0 <= self.vaccination_pct <= 100.0):
            raise ValueError("Vaccination percentage must be between 0 and 100.")

    @property
    def basic_reproduction_number(self) -> float:
        """R₀ = beta / gamma — threshold for epidemic spread."""
        return self.beta / self.gamma

    def get_initial_conditions(self) -> Tuple[float, float, float]:
        """
        Compute initial S0, I0, R0 after applying vaccination.

        Vaccinated individuals are moved from S → R (immune) at t=0.
        """
        S0 = float(self.N - self.I0 - self.R0_initial)
        I0 = float(self.I0)
        R0 = float(self.R0_initial)

        if self.vaccination_pct > 0:
            vaccinated = (self.vaccination_pct / 100.0) * self.N
            vaccinated = min(vaccinated, S0)  # Cannot vaccinate more than are susceptible
            S0 -= vaccinated
            R0 += vaccinated

        return S0, I0, R0


@dataclass
class SIRResult:
    """Output of a single SIR simulation run."""
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    N: int
    vaccination_pct: float = 0.0

    @property
    def peak_infected(self) -> float:
        return float(np.max(self.I))

    @property
    def peak_day(self) -> int:
        return int(np.argmax(self.I))

    @property
    def final_recovered(self) -> float:
        return float(self.R[-1])

    @property
    def final_susceptible(self) -> float:
        return float(self.S[-1])

    def verify_conservation(self, tol: float = 1.0) -> bool:
        """Verify S + I + R ≈ N at all time steps (conservation check)."""
        total = self.S + self.I + self.R
        return bool(np.allclose(total, self.N, atol=tol))


def _sir_odes(y: list, t: float, N: int, beta: float, gamma: float) -> list:
    """
    SIR ordinary differential equations.

    dS/dt = -beta * S * I / N
    dI/dt =  beta * S * I / N - gamma * I
    dR/dt =  gamma * I

    Parameters
    ----------
    y     : [S, I, R] — current state
    t     : current time (unused directly, required by odeint)
    N     : total population
    beta  : transmission rate
    gamma : recovery rate
    """
    S, I, R = y
    force_of_infection = beta * S * I / N
    dS = -force_of_infection
    dI = force_of_infection - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def run_simulation(params: SIRParameters, vaccination_pct: float = None) -> SIRResult:
    """
    Run a single SIR simulation.

    Parameters
    ----------
    params          : validated SIRParameters object
    vaccination_pct : override vaccination percentage (uses params value if None)

    Returns
    -------
    SIRResult with time series arrays t, S, I, R
    """
    # Optionally override vaccination for comparison runs
    if vaccination_pct is not None:
        # Temporarily patch vaccination without mutating original
        override = SIRParameters(
            N=params.N,
            I0=params.I0,
            R0_initial=params.R0_initial,
            beta=params.beta,
            gamma=params.gamma,
            days=params.days,
            vaccination_pct=vaccination_pct,
        )
        S0, I0, R0 = override.get_initial_conditions()
    else:
        S0, I0, R0 = params.get_initial_conditions()
        vaccination_pct = params.vaccination_pct

    # Time grid: 1000 points for smooth curves
    t = np.linspace(0, params.days, 1000)

    # Integrate
    solution = odeint(
        _sir_odes,
        y0=[S0, I0, R0],
        t=t,
        args=(params.N, params.beta, params.gamma),
    )

    S, I, R = solution[:, 0], solution[:, 1], solution[:, 2]

    # Clip numerical noise below 0
    S = np.clip(S, 0, params.N)
    I = np.clip(I, 0, params.N)
    R = np.clip(R, 0, params.N)

    return SIRResult(t=t, S=S, I=I, R=R, N=params.N, vaccination_pct=vaccination_pct)
