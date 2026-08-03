# Quadruped Gait Simulator

A kinematic quadruped that measures its own static stability, and reproduces the
three quarter duty factor threshold of McGhee and Frank (1968) by measurement.

[![CI](https://github.com/Eelis03/quadruped-gait-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/quadruped-gait-simulator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Minimum static stability margin of a lateral sequence walk plotted against duty factor, falling almost linearly to exactly zero at a duty factor of three quarters and rising again above it, with a second panel showing the fraction of the cycle that has a support polygon reaching one at that same duty factor](docs/figures/duty_factor_sweep.png)

## The result

McGhee and Frank proved in 1968 that a quadruped creeping gait cannot be
statically stable below a duty factor of three quarters. This repository derives
nothing analytically. It builds the gait, places the feet, computes the convex
hull of the loaded ones, measures the distance from the projected centre of mass
to its boundary, and reports the smallest value it saw.

The measurement lands on the analytical answer. Sweeping the duty factor of a
lateral sequence walk in steps of 0.0125 from 0.600 to 0.900, holding everything
else fixed, the minimum static margin is

```
   beta   support    ssm_min   ssm_mean    lsm_min   area_mean   stable
  0.738     0.960    0.00250    0.05323    0.00317     0.15663    False
  0.750     1.000    0.00000    0.05306   -0.00000     0.15665    False
  0.762     1.000    0.00452    0.06464    0.00573     0.16357     True
```

The margin is zero at 0.750 and nowhere else in the sweep. Below it the margin is
positive but the gait spends part of the cycle on two feet, so for that part
there is no support polygon to be inside and the criterion certifies nothing.
Above it the margin is strictly positive and the criterion certifies the gait.
The threshold is a single point, and it is the published one.

Zero here is not a rounding artefact. It is a geometric coincidence that a 0.06 m
lateral trunk sway does not remove, and the reason is visible rather than
algebraic:

![Four support polygons of a walk at duty factor 0.75 spanning one support transition, the projected centre of mass moving toward the shared diagonal edge, lying exactly on it at the instant the margin reads zero, then moving away inside the next support triangle](docs/figures/critical_support_polygon.png)

At a duty factor of exactly 0.75 the swing windows tile the cycle without overlap
and without gap, so one leg lifts at the instant another lands and there is never
a fourth foot down. The outgoing support triangle and the incoming one therefore
share an edge, and that shared edge is the line the centre of mass is crossing at
that instant. No lateral sway fixes this, because the sign of the offset that
would help reverses at the very moment the edge becomes critical.

This is also why the trot, the pace and the bound return `n/a` rather than a
number. None of them ever has three feet on the ground, so none of them has a
support polygon, and a quasi-static criterion has nothing to say about them. That
is the honest answer, not a gap.

## The model that produces it

Three pieces, layered so that each can be checked on its own.

**Legs.** A hip roll joint about the trunk x axis, a fixed lateral offset, then
hip pitch and knee pitch about parallel y axes, which is the layout of ANYmal
(Hutter et al., 2016) and of the MIT Cheetah family (Bledt et al., 2018). The
inverse is closed form: the lateral components of the target fix the roll angle,
the planar two link subchain that remains falls to the law of cosines and one two
argument arctangent (Siciliano et al., 2009), and two branch choices make the
solution single valued. Targets outside the annulus spanned by the links, or
inside the cylinder swept by the abduction offset, are reported as unreachable
rather than clipped.

**Gaits.** A period, a duty factor, and one phase offset per leg, in the
parameterisation of Hildebrand (1965, 1989). Leg `i` is loaded while
`(t / T - phi_i) mod 1 < beta`. That is a periodic step function, so the time a
leg spends loaded over any window is an integral rather than a tally, and the
realised duty factor this repository reports is computed in closed form and
carries no discretisation error at any sampling rate. Footholds go at the neutral
point under the hip at mid stance, which is Raibert's symmetry condition
(Raibert, 1986). Swing feet follow a cycloid or a quintic Bezier with vertical
end control edges; stance feet are fixed in the world, so all their motion
relative to the trunk is the commanded body twist and the simulator needs no
integration step.

**Stability.** The support polygon is the convex hull of the loaded feet, from
Qhull through SciPy (Barber et al., 1996). The static stability margin is the
shortest distance from the vertically projected centre of mass to its boundary,
positive inside and negative outside. The longitudinal margin is the same
quantity along the direction of travel, obtained by clipping the line through the
centre of mass against the polygon half planes. Both are `nan` when fewer than
three feet are loaded or the loaded feet are collinear. Definitions and sign
convention are from McGhee and Frank (1968).

The alternatives weighed and rejected, including the zero moment point, the
energy stability margin of Messuri and Klein (1985), and a full rigid body
simulation, are recorded in [docs/design-notes.md](docs/design-notes.md) next to
the method each was weighed against.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/quadruped-gait-simulator.git
cd quadruped-gait-simulator
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The package ships a `py.typed` marker, so an installing project gets the
annotations rather than `Any`.

Simulate the reference walk and read off its stability:

```python
from quadruped_gait import format_report, simulate, summarise
from quadruped_gait.pipeline import reference_walk

trace = simulate(reference_walk(cycles=3.0, samples_per_cycle=200))
report = summarise(trace)

print(report.stability.minimum_static)         # 0.01233... metres
print(report.stability.statically_stable)      # True
print(report.contact.exact_duty_factors[0])    # 0.8000000000000002
print(format_report(report))
```

The last line is the closed form realised duty factor. It is the commanded 0.80
to within floating point rounding, and it stays that way whatever the sampling
rate is set to.

Build a gait by hand and ask the schedule a question it can answer exactly:

```python
from quadruped_gait import gait, stance_count_extrema
from quadruped_gait.model import LegId

trot = gait("trot", period=0.5, duty_factor=0.5)
print(trot.contact_state(0.125).stance_count)          # 2
print(stance_count_extrema(trot))                      # (2, 2)
print(trot.stance_intervals(LegId.FRONT_LEFT, 0.0, 1.0))
# ((0.0, 0.25), (0.5, 0.75))
print(trot.stance_fraction(LegId.FRONT_LEFT, 0.0, 1.0))  # 0.5 exactly
```

Solve the leg inverse kinematics directly:

```python
import math

from quadruped_gait import default_robot, inverse_kinematics
from quadruped_gait.model import LegId

robot = default_robot()
target = robot.nominal_foot_in_hip(LegId.FRONT_LEFT)
angles = inverse_kinematics(robot.leg, target, lateral_sign=1.0)
print(math.degrees(angles.knee_pitch))             # -91.1459...
```

## Results

Every number below is printed by the command shown above it, on the reference
robot: 0.50 m between the fore and hind hip axes, 0.30 m between the left and
right hip axes, a 0.08 m abduction offset, 0.30 m thigh and shank links, a 0.42 m
standing trunk height, and the centre of mass at the trunk origin.

### The four library gaits

`uv run python examples/gait_comparison.py`, three cycles at 0.30 m/s with 200
samples per cycle and no lateral trunk offset. `ssm` is the static stability
margin and `lsm` the longitudinal stability margin, both in metres.

```
gait      beta  beta_real      err  support   ssm_min  ssm_mean   lsm_min   feet
--------------------------------------------------------------------------------
walk     0.750     0.7500   0.0000    1.000    0.0000    0.0275   -0.0000   3.00
trot     0.500     0.5000   0.0000    0.000       n/a       n/a       n/a   2.00
pace     0.500     0.5000   0.0000    0.000       n/a       n/a       n/a   2.00
bound    0.400     0.4000   0.0000    0.000       n/a       n/a       n/a   1.60
```

`beta_real` and `feet` are closed form, so `err` stays at floating point zero for
every gait at every sampling rate. The contact schedules that produce those
columns:

![Contact schedules of walk, trot, pace and bound over two gait cycles, one horizontal bar per leg, showing the walk swing windows tiling the cycle without overlap so three feet are always down, the trot and pace holding exactly two, and the bound leaving two intervals with no foot on the ground](docs/figures/gait_diagrams.png)

The walk swing windows tile the cycle, which is why exactly one leg is ever
airborne and why its panel reports `3 to 3 feet down`. The trot and the pace hold
two feet at every instant. The bound averages 1.60 feet and reports `0 to 2`, so
part of every cycle has no foot on the ground at all. Only the walk ever forms a
support polygon.

`uv run python examples/gait_diagram.py` renders the same schedules as text, one
cycle at 60 columns. The walk:

```
cycle length 1.000 s, one column is 0.0167 s
FL |#############################################...............|
FR |###############...............##############################|
HL |##############################...............###############|
HR |...............#############################################|
    0                                                          T
```

and the bound, where the two gaps down every column are the aerial phases:

```
cycle length 0.400 s, one column is 0.0067 s
FL |########################....................................|
FR |########################....................................|
HL |..............................########################......|
HR |..............................########################......|
    0                                                          T
```

### The duty factor sweep

Same command, second table, sweeping the walk duty factor with a 0.06 m lateral
trunk offset. The figure at the top of this page is the same experiment at 25
points instead of 7, written by `uv run python examples/docs_figures.py`.

```
   beta   support    ssm_min   ssm_mean    lsm_min   area_mean   stable
-----------------------------------------------------------------------
  0.600     0.400    0.02960    0.05230    0.03778     0.15399    False
  0.650     0.607    0.01989    0.05340    0.02531     0.15545    False
  0.700     0.800    0.01000    0.05362    0.01269     0.15634    False
  0.750     1.000    0.00000    0.05306   -0.00000     0.15665    False
  0.800     1.000    0.01234    0.09115    0.01566     0.17972     True
  0.850     1.000    0.02224    0.12531    0.02829     0.20141     True
  0.900     1.000    0.03194    0.15758    0.04078     0.22303     True
```

`stable` requires every sample to have a support polygon and a positive margin.
Below 0.750 the `support` column shows the cycle spending time on two feet.
At 0.750 the support never drops but the margin touches zero. Above it, an
interval of four foot support appears at each transition and the minimum margin
grows close to linearly with the excess duty factor, from 0.01234 m at 0.800 to
0.03194 m at 0.900.

### The walk sitting on the threshold

`uv run python examples/docs_figures.py`, the walk of the figure above, three
cycles at a duty factor of exactly 0.75 with the 0.06 m lateral trunk offset:

```
commanded duty     0.7500
realised duty      FL=0.7500  FR=0.7500  HL=0.7500  HR=0.7500
duty error         0.00e+00
support histogram  0feet=0  1feet=0  2feet=0  3feet=600  4feet=0
feet down range    3 to 3
mean stance feet   3.0000
supported fraction 1.0000
static margin      min 0.0000 m, mean 0.0531 m
longitudinal       min -0.0000 m, mean 0.0673 m
support area       mean 0.1566 m^2
statically stable  False
```

Three feet are loaded at all 600 samples and the mean support area is 0.1566
square metres, yet the run is not certified, because the minimum margin over the
cycle is zero. A gait can be supported everywhere and stable nowhere.

### The reference walk, which is certified

`uv run python examples/walk_stability.py`, the same walk at a duty factor of
0.80, three cycles at 0.30 m/s, 200 samples per cycle:

```
commanded duty     0.8000
realised duty      FL=0.8000  FR=0.8000  HL=0.8000  HR=0.8000
duty error         1.11e-16
sampled duty       FL=0.8000  FR=0.8000  HL=0.8000  HR=0.8017
sampling error     1.67e-03
support histogram  0feet=0  1feet=0  2feet=0  3feet=479  4feet=121
feet down range    3 to 4
mean stance feet   3.2000
sampled stance     3.2017
unreachable        0
samples            600
supported fraction 1.0000
static margin      min 0.0123 m, mean 0.0911 m
longitudinal       min 0.0157 m, mean 0.1059 m
support area       mean 0.1797 m^2
statically stable  True
```

The `sampled` rows are the same quantities counted from the recorded trace, kept
as a cross check. Their gap from the closed form values, 1.67e-03 here, is the
whole discretisation error of a 200 sample cycle, and it is the number the
reported duty factor used to carry. All 2400 commanded foot positions are inside
the leg workspace, so the inverse kinematics never fails.

### Inverse kinematics accuracy

`uv run python examples/leg_kinematics.py`, 20000 random joint triples per leg
drawn from the limits `((-0.7, 0.7), (-0.55, 1.2), (-1.7, -0.3))` radians, mapped
to foot positions by forward kinematics and solved back:

| Leg | Targets | Solved | Max error | Mean error | Median error |
| --- | --- | --- | --- | --- | --- |
| front left | 20000 | 20000 | 2.618e-16 m | 6.726e-17 m | 6.206e-17 m |
| front right | 20000 | 20000 | 2.884e-16 m | 7.104e-17 m | 6.206e-17 m |

The reconstruction is exact to floating point rounding. The nominal standing
posture of the front left leg is hip roll 0.0000 deg, hip pitch +45.5730 deg,
knee pitch -91.1460 deg, which matches the isosceles triangle formed by two equal
0.300 m links spanning a 0.420 m drop. A target 0.650 m from the hip pitch axis
is rejected with the message `target is 0.650000 m from the hip pitch axis,
beyond the maximum reach of 0.600000 m`.

## What this model cannot tell you

It is kinematic and quasi-static. It integrates no rigid body dynamics and
computes no contact forces. Everything above is a statement about geometry.

- A gait it calls statically stable is not guaranteed to keep the robot upright.
  Falling is a dynamic event, and this criterion assumes flat rigid ground,
  unlimited friction, and negligible inertia.
- A gait it reports as `n/a` is not being called unstable. It is being reported
  as outside the reach of the criterion.
- No torque, joint velocity, or joint limit is checked, so a trace is not proof
  that an actuator could follow it.
- The 0.75 threshold is a property of this symmetric machine with its centre of
  mass at the trunk origin. The mechanism generalises; the number does not.

[docs/design-notes.md](docs/design-notes.md) states the rest of the limitations
precisely, and records which one was closed, what closing it cost, and what is
left of it.

## Verification

```bash
uv run pytest -q
uv run pytest --cov=src/quadruped_gait --cov-report=term-missing
uv run ruff check .
uv run mypy
```

311 tests pass in about 15 seconds. Statement coverage of `src/quadruped_gait` is
98.43 percent, 19 uncovered statements out of 1207. CI runs the coverage command
above with `--cov-fail-under=96` and fails the build below that. `mypy` runs in
strict mode over the whole package, on both Linux and Windows.

The suite has three tiers. The property and invariant tier asserts that forward
kinematics composed with the inverse returns the original foot position, that
unreachable targets are reported for all three failure modes, that each gait is
periodic with its configured period, that the closed form stance measure is
additive over a split window and invariant under a whole cycle shift, that
refining the sampling rate drives the counted duty factor onto the closed form
one, that a trot has exactly two feet in stance and a walk at least three, that
the support polygon is empty below three non-collinear contacts, that loaded feet
neither slip nor leave the ground plane, and that the sign convention of both
margins matches values computed by hand on a unit square and on a right triangle.

The regression tier recomputes the reference walk and compares the trunk pose,
foot positions, joint angles, contact flags, and both stability margins against
`tests/data/reference_walk.json` with a tolerance of 1e-9. Run
`uv run python tests/test_regression.py` to regenerate that file after a reviewed
change of behaviour.

The integration tier loads every script in `examples/`, runs it under a reduced
sample count, and checks that it exits cleanly and writes what it says it writes.

### About the figures

The three images in `docs/figures/` are committed snapshots, not build artefacts.
Regenerate them with one command:

```bash
uv run python examples/docs_figures.py
```

That is the only script that writes into a tracked directory. The other examples
write scratch figures into `figures/`, which is not tracked.

CI runs that command on both Linux and Windows to prove it still works, and
writes the result to a temporary directory. It does not compare the output
against the committed files byte for byte, because Matplotlib output is not byte
reproducible across platforms: font availability, FreeType version, and PNG
encoder details all shift the bytes without changing the picture. The same
command prints the size of what it wrote, `total 161.7 kB across 3 figures`,
which is the result of a deliberate choice of figure dimensions and 110 dots per
inch. No compression dependency is involved.

## Module layout

| Module | Responsibility |
| --- | --- |
| `src/quadruped_gait/model/geometry.py` | Leg link lengths, trunk hip layout, nominal standing posture, leg ordering |
| `src/quadruped_gait/model/transforms.py` | Rotation matrices, roll-pitch-yaw conversion, and the SE(3) trunk pose |
| `src/quadruped_gait/model/kinematics.py` | Forward kinematics, closed form inverse kinematics, reachability and branch tests |
| `src/quadruped_gait/model/contact.py` | Contact state of the four feet and its derived views |
| `src/quadruped_gait/model/workspace.py` | Workspace sampling and forward-inverse round trip error measurement |
| `src/quadruped_gait/algorithm/gait.py` | Duty factor and phase offset scheduler, closed form stance measure and intervals, gait library |
| `src/quadruped_gait/algorithm/swing.py` | Cycloidal and Bezier swing trajectories, de Casteljau evaluation, stance constraint |
| `src/quadruped_gait/algorithm/stability.py` | Convex hull support polygon, static and longitudinal stability margins |
| `src/quadruped_gait/pipeline/simulator.py` | Closed form trunk motion, foothold planning, and the recorded trace |
| `src/quadruped_gait/pipeline/sweep.py` | Batch runs over a parameter axis, used for the duty factor sweep |
| `src/quadruped_gait/pipeline/presets.py` | Named configurations shared by the examples and the regression test |
| `src/quadruped_gait/analysis/metrics.py` | Contact statistics, stability series and summaries, gait diagram intervals |
| `src/quadruped_gait/analysis/report.py` | Fixed width text rendering of every result table |
| `src/quadruped_gait/analysis/figures.py` | Duty sweep, gait diagram, stability history, foot path, and support polygon figures |
| `examples/` | Argument parsing and calls into the library, with no computation of their own |

Layers depend downward only. `model` imports nothing from the package,
`algorithm` imports `model`, `pipeline` imports both, `analysis` imports all
three. Nothing in `model` or `algorithm` performs input or output, and no module
other than `analysis/figures.py` imports Matplotlib.

## References

### Gait definitions and stability criteria

- Hildebrand, M. (1965). Symmetrical gaits of horses. *Science* 150(3697),
  701-708. DOI: [10.1126/science.150.3697.701](https://doi.org/10.1126/science.150.3697.701).
  Source of the duty factor and relative phase parameterisation, and of the
  footfall orders used for the walk, trot, and pace.
- Hildebrand, M. (1989). The quadrupedal gaits of vertebrates. *BioScience*
  39(11), 766-775. DOI: [10.2307/1311182](https://doi.org/10.2307/1311182).
  Source of the lateral sequence walk and of the transverse bound pattern.
- McGhee, R. B. and Frank, A. A. (1968). On the stability properties of
  quadruped creeping gaits. *Mathematical Biosciences* 3, 331-351. DOI:
  [10.1016/0025-5564(68)90090-4](https://doi.org/10.1016/0025-5564%2868%2990090-4).
  Source of the static and longitudinal stability margins, of the sign
  convention, and of the three quarter duty factor threshold this repository
  reproduces by measurement.
- Messuri, D. A. and Klein, C. A. (1985). Automatic body regulation for
  maintaining stability of a legged vehicle during rough-terrain locomotion.
  *IEEE Journal on Robotics and Automation* 1(3), 132-141. DOI:
  [10.1109/JRA.1985.1087012](https://doi.org/10.1109/JRA.1985.1087012). Source
  of the energy stability margin, which is discussed and rejected in the design
  notes.
- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press. ISBN:
  978-0-262-18117-4. Source of the neutral point foothold rule that places each
  stance symmetrically about the hip.
- Song, S.-M. and Waldron, K. J. (1989). *Machines That Walk: The Adaptive
  Suspension Vehicle*. MIT Press. ISBN: 978-0-262-19274-3. Source of the wave
  gait analysis and of the lateral trunk offset used to enlarge the margin of a
  crawl gait.

### Kinematics and trajectory generation

- Siciliano, B., Sciavicco, L., Villani, L. and Oriolo, G. (2009). *Robotics:
  Modelling, Planning and Control*. Springer. DOI:
  [10.1007/978-1-84628-642-1](https://doi.org/10.1007/978-1-84628-642-1).
  Source of the roll-pitch-yaw convention and of the planar two link inverse
  kinematics solution the leg solver is built on.
- Hutter, M., Gehring, C., Jud, D., Lauber, A., Bellicoso, C. D., Tsounis, V.,
  Hwangbo, J., Bodie, K., Fankhauser, P., Bloesch, M., Diethelm, R., Bachmann,
  S., Melzer, A. and Hoepflinger, M. (2016). ANYmal: a highly mobile and dynamic
  quadrupedal robot. *IEEE/RSJ International Conference on Intelligent Robots
  and Systems*, 38-44. DOI:
  [10.1109/IROS.2016.7758092](https://doi.org/10.1109/IROS.2016.7758092).
  Source of the three degree of freedom leg layout and of the link proportions
  the reference robot approximates.
- Bledt, G., Powell, M. J., Katz, B., Di Carlo, J., Wensing, P. M. and Kim, S.
  (2018). MIT Cheetah 3: design and control of a robust, dynamic quadruped
  robot. *IEEE/RSJ International Conference on Intelligent Robots and Systems*,
  2245-2252. DOI:
  [10.1109/IROS.2018.8593885](https://doi.org/10.1109/IROS.2018.8593885).
  Source of the Bezier swing foot parameterisation and of the practice of
  driving a quadruped from a duty factor and a phase offset vector.
- Winkler, A. W., Bellicoso, C. D., Hutter, M. and Buchli, J. (2018). Gait and
  trajectory optimization for legged systems through phase-based end-effector
  parameterization. *IEEE Robotics and Automation Letters* 3(3), 1560-1567. DOI:
  [10.1109/LRA.2018.2798285](https://doi.org/10.1109/LRA.2018.2798285). Source
  of the phase based split of each foot into alternating stance and swing
  segments, which is the structure of the scheduler used here.
- Farin, G. (2002). *Curves and Surfaces for CAGD: A Practical Guide*, 5th
  edition. Morgan Kaufmann. ISBN: 978-1-55860-737-8. Source of the de Casteljau
  algorithm used to evaluate the Bezier swing curve.

### Computational geometry

- Barber, C. B., Dobkin, D. P. and Huhdanpaa, H. (1996). The Quickhull algorithm
  for convex hulls. *ACM Transactions on Mathematical Software* 22(4), 469-483.
  DOI: [10.1145/235815.235821](https://doi.org/10.1145/235815.235821). The
  algorithm behind `scipy.spatial.ConvexHull`, which builds the support polygon.
- de Berg, M., Cheong, O., van Kreveld, M. and Overmars, M. (2008).
  *Computational Geometry: Algorithms and Applications*, 3rd edition. Springer.
  DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2).
  Source of the half plane clipping used for the longitudinal margin and of the
  point in convex polygon test.

### Dependencies

| Package | Purpose | Licence |
| --- | --- | --- |
| [NumPy](https://numpy.org/) >= 2.0 | Array storage for traces, and the vector and matrix algebra in the kinematics and the geometry | BSD-3-Clause |
| [SciPy](https://scipy.org/) >= 1.14 | `scipy.spatial.ConvexHull`, the Qhull binding that builds the support polygon | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org/) >= 3.9 | Every figure in this README | Matplotlib licence, a BSD compatible PSF style licence |
| [pytest](https://pytest.org/) >= 8.3 | Test runner for all three test tiers | MIT |
| [pytest-cov](https://pytest-cov.readthedocs.io/) >= 6.0 | Coverage measurement and the CI coverage gate | MIT |
| [Ruff](https://docs.astral.sh/ruff/) >= 0.8 | Linting and import ordering | MIT |
| [mypy](https://mypy-lang.org/) >= 1.13 | Static type checking of `src/quadruped_gait` under strict settings | MIT |

Dependency citations:

- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585,
  357-362. DOI:
  [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific
  computing in Python. *Nature Methods* 17, 261-272. DOI:
  [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
- Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. *Computing in
  Science and Engineering* 9(3), 90-95. DOI:
  [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

## License

Released under the MIT license. See [LICENSE](LICENSE).
