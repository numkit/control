# numkit/control

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Control System Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance linear time-invariant (LTI) modeling, analysis, and control design algorithms with MATLAB R2025b parity.

---

## Features

- **LTI Model Creation & Data Access**: Transfer functions (`tf`, `filt`), zero-pole-gain (`zpk`), state-space (`ss`), frequency-response data (`frd`), and extraction (`tfdata`, `zpkdata`, `ssdata`, `frdata`).
- **Model Interconnections & Realizations**: `series`, `parallel`, `feedback`, state coordinate transformations (`ss2ss`), and minimal realizations (`minreal`).
- **System Properties & Poles/Zeros**: `pole`, `zero`, `tzero`, `damp`, `pzmap`, `order`, `isct`, `isdt`, `issiso`, `isproper`, `isstable`, `isstatic`.
- **Time & Frequency Response**: `step`, `impulse`, `initial`, `lsim`, `evalfr`, `freqresp`, `bode`, `nyquist`, `rlocus`.
- **Stability & Performance Analysis**: `dcgain`, `margin`, `allmargin`, `stepinfo`, $H_\infty$ norm (`hinfnorm`).
- **Discretization**: Continuous/discrete conversion (`c2d`, `d2c`) supporting ZOH, FOH, Tustin, prewarp, and impulse-invariant methods.
- **State-Space & Optimal Control**: Controllability/observability (`ctrb`, `obsv`), Gramians (`gram`), covariance (`covar`), pole placement (`acker`, `place`), Lyapunov equations (`lyap`, `dlyap`), algebraic Riccati equations (`care`, `dare`), and LQR design (`lqr`, `dlqr`).

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `numkit_control.dll` (Windows) or `libnumkit_control.so` (Linux/macOS).

---

## Usage in NumKit

Install via the NumKit package manager:

```bash
nk pkg install control
```

Or load dynamically in a script:

```matlab
% Loaded automatically on first call to any control function:
sys = tf([1], [1 2 1]);
[A, B, C, D] = ssdata(ss(sys));
disp(ctrb(A, B));
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
