# Quadruped Gait Simulator

Trot and walk gait generation for a quadruped with support polygon and stability margin analysis.

[![CI](https://github.com/Eelis03/quadruped-gait-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/quadruped-gait-simulator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library generates periodic gaits for a four legged robot and measures how
statically stable they are. It contains three degree of freedom leg kinematics
with a closed form inverse, a gait scheduler covering walk, trot, pace, and
bound, swing and stance foot trajectory generation, and the support polygon and
stability margin analysis of McGhee and Frank (1968). It is intended for
engineers who need to compare contact schedules and foot placement strategies
before committing to a dynamics simulator or to hardware.

The model is kinematic and quasi-static throughout. It integrates no rigid body
dynamics and computes no contact forces. The file
[docs/design-notes.md](docs/design-notes.md) states precisely which conclusions
therefore do not follow from it.

## Problem

A legged robot has to decide, at every instant, which feet carry load and where
the swinging feet will land. Those two decisions fix a support polygon, and the
classical quasi-static criterion for a slow gait is that the vertical projection
of the centre of mass stays inside that polygon with a margin. The engineering
questions that follow are concrete:

1. Given link lengths and a commanded trunk pose, is a requested foot position
   reachable, and what joint angles achieve it?
2. Given a duty factor and a set of leg phase offsets, what contact schedule
   results, and does the realised stance fraction match what was commanded?
3. Given that schedule and a commanded body velocity, where do the feet land,
   what support polygon do they form, and how large is the stability margin?
4. Which gaits does the quasi-static criterion certify at all, and which does it
   say nothing about?

The last question matters because a trot and a bound never place three feet on
the ground at once. Any answer that reports a positive static margin for them is
wrong, and the code has to say so rather than return a number.

## Approach

Legs are modelled as a hip roll joint about the trunk x axis followed by a fixed
lateral offset and two parallel pitch joints, the layout used by ANYmal (Hutter
et al., 2016) and by the MIT Cheetah family (Bledt et al., 2018). Forward
kinematics is a direct composition of those transforms. The inverse is closed
form: the lateral components of the target fix the roll angle, the planar two
link subchain that remains is solved by the law of cosines and a single two
argument arctangent following Siciliano et al. (2009), and two branch choices,
the knee sign and the side of the hip axis the foot lies on, make the solution
single valued. Targets outside the annulus spanned by the links, or inside the
cylinder swept by the abduction offset, are reported as unreachable rather than
clipped.

Gaits are parameterised in the form of Hildebrand (1965, 1989): a period, a duty
factor, and one phase offset per leg. The four library gaits use the standard
footfall patterns, a lateral sequence walk with quarter cycle spacing, diagonal
pairs for the trot, lateral pairs for the pace, and fore against hind for the
bound. Footholds are placed at the neutral point under the hip at mid stance,
which is Raibert's symmetry condition for steady locomotion (Raibert, 1986).
Swing feet follow either a cycloidal profile or a quintic Bezier whose end
control edges are vertical, and stance feet are fixed in the world, so all of
their motion relative to the trunk is the commanded body twist.

Stability is measured as in McGhee and Frank (1968). The support polygon is the
convex hull of the loaded feet, computed with Qhull through SciPy (Barber et
al., 1996). The static stability margin is the shortest distance from the
vertically projected centre of mass to the polygon boundary, positive inside and
negative outside. The longitudinal stability margin is the same quantity
measured only along the direction of travel, obtained by clipping the line
through the centre of mass against the polygon half planes. Both are reported as
undefined when fewer than three feet are loaded or the loaded feet are
collinear.

The alternatives that were considered and rejected, including a zero moment
point criterion, an energy stability margin, and a full rigid body simulation,
are recorded in [docs/design-notes.md](docs/design-notes.md).

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

## Usage

Simulate the reference walk and read off its stability:

```python
from quadruped_gait import format_report, simulate, summarise
from quadruped_gait.pipeline import reference_walk

trace = simulate(reference_walk(cycles=3.0, samples_per_cycle=200))
report = summarise(trace)

print(report.stability.minimum_static)     # 0.01233... metres
print(report.stability.statically_stable)  # True
print(format_report(report))
```

Build a gait by hand and inspect one instant of its contact schedule:

```python
from quadruped_gait import gait

trot = gait("trot", period=0.5, duty_factor=0.5)
state = trot.contact_state(0.125)
print(state.stance_count)                          # 2
print([leg.name for leg in state.stance_legs])     # ['FRONT_LEFT', 'HIND_RIGHT']
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

Runnable examples live in `examples/`:

```bash
uv run python examples/leg_kinematics.py
uv run python examples/gait_diagram.py
uv run python examples/walk_stability.py
uv run python examples/gait_comparison.py
```

Each script accepts `--no-figures` and a reduced `--samples-per-cycle` for quick
runs. Figures are written to `figures/`, which is not tracked.

## Results

All numbers below are the output of the commands shown, on the reference robot:
0.50 m between the fore and hind hip axes, 0.30 m between the left and right hip
axes, a 0.08 m abduction offset, 0.30 m thigh and shank links, a 0.42 m standing
trunk height, and the centre of mass at the trunk origin.

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
knee pitch -91.1460 deg, which matches the isosceles triangle formed by two
equal 0.300 m links spanning a 0.420 m drop. A target 0.650 m from the hip pitch
axis is rejected with the message `target is 0.650000 m from the hip pitch axis,
beyond the maximum reach of 0.600000 m`.

### Gait comparison

`uv run python examples/gait_comparison.py`, three cycles at 0.30 m/s with 200
samples per cycle and no lateral trunk offset. `ssm` is the static stability
margin and `lsm` the longitudinal stability margin, both in metres. A value of
`n/a` means no sample in the run had three or more non-collinear feet on the
ground.

```
gait      beta  beta_real      err  support   ssm_min  ssm_mean   lsm_min   feet
--------------------------------------------------------------------------------
walk     0.750     0.7500   0.0000    1.000    0.0000    0.0275   -0.0000   3.00
trot     0.500     0.5000   0.0000    0.000       n/a       n/a       n/a   2.00
pace     0.500     0.5000   0.0000    0.000       n/a       n/a       n/a   2.00
bound    0.400     0.4042   0.0050    0.000       n/a       n/a       n/a   1.62
```

Three results follow from this table.

The realised duty factor matches the command to within one sampling interval.
The walk, trot, and pace reproduce it exactly at 200 samples per cycle. The
bound is off by 0.0050, which is exactly one sample out of 200. Raising the rate
to 1000 samples per cycle drops the largest per leg error to at most 0.0015 for
every gait, which is the tolerance asserted in the test suite.

The trot and the pace hold exactly two feet on the ground at every instant, and
the bound averages 1.62 feet with an aerial phase covering 19 percent of the
cycle. None of them ever forms a support polygon, so the quasi-static criterion
returns `n/a` rather than a margin. That is the correct answer, not a gap: these
gaits are stabilised by momentum, and nothing in this model can certify them.

The lateral sequence walk at a duty factor of 0.750 keeps exactly three feet
loaded for the whole cycle, but its minimum static margin is 0.0000 m. At each of
the four support transitions the critical diagonal edge of the outgoing support
triangle and of the incoming one is the same line, and that line passes through
the centre of mass. No lateral trunk offset removes this, because the required
sign of the offset reverses at the very instant the edge is critical.

### Walk duty factor sweep

Same command, second table, sweeping the walk duty factor with a 0.06 m lateral
trunk offset:

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

The `stable` column requires every sample to have a support polygon and a
positive margin. It flips at a duty factor of 0.75, which is the threshold
McGhee and Frank (1968) derive analytically for a quadruped creeping gait. Below
0.75 the support drops to two feet for part of the cycle, so the `support`
fraction falls below one. At exactly 0.75 the support never drops but the margin
touches zero. Above 0.75 an interval of four foot support appears at each
transition and the margin becomes strictly positive. The minimum static margin
then grows close to linearly with the excess duty factor, from 0.01234 m at
0.800 to 0.03194 m at 0.900.

### Reference walk

`uv run python examples/walk_stability.py`, three cycles of a lateral sequence
walk at a duty factor of 0.80 with a 0.06 m lateral trunk offset, driven at
0.30 m/s, 200 samples per cycle:

```
gait               walk
period             1.000 s
forward velocity   0.300 m/s
stride length      0.300 m
commanded duty     0.8000
realised duty      FL=0.8000  FR=0.8000  HL=0.8000  HR=0.8017
duty error         1.67e-03
support histogram  0feet=0  1feet=0  2feet=0  3feet=479  4feet=121
mean stance feet   3.2017
unreachable        0
samples            600
supported fraction 1.0000
static margin      min 0.0123 m, mean 0.0911 m
longitudinal       min 0.0157 m, mean 0.1059 m
support area       mean 0.1797 m^2
statically stable  True
```

Every one of the 2400 commanded foot positions is inside the leg workspace, so
the inverse kinematics never fails. The mean number of loaded feet, 3.2017,
matches four times the duty factor, 3.200, to within one sample.

### Contact schedules

`uv run python examples/gait_diagram.py` renders one cycle per gait, `#` for
stance and `.` for swing, 60 columns:

```
walk, cycle length 1.000 s
FL |#############################################...............|
FR |###############...............##############################|
HL |##############################...............###############|
HR |...............#############################################|

trot, cycle length 0.500 s
FL |##############################..............................|
FR |..............................##############################|
HL |..............................##############################|
HR |##############################..............................|

pace, cycle length 0.500 s
FL |##############################..............................|
FR |..............................##############################|
HL |##############################..............................|
HR |..............................##############################|

bound, cycle length 0.400 s
FL |########################....................................|
FR |########################....................................|
HL |..............................########################......|
HR |..............................########################......|
```

The walk swing windows tile the cycle without overlap, which is why exactly one
leg is ever airborne. The bound leaves two intervals with no leg on the ground.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/quadruped_gait/model/geometry.py` | Leg link lengths, trunk hip layout, nominal standing posture, leg ordering |
| `src/quadruped_gait/model/transforms.py` | Rotation matrices, roll-pitch-yaw conversion, and the SE(3) trunk pose |
| `src/quadruped_gait/model/kinematics.py` | Forward kinematics, closed form inverse kinematics, reachability and branch tests |
| `src/quadruped_gait/model/contact.py` | Contact state of the four feet and its derived views |
| `src/quadruped_gait/model/workspace.py` | Workspace sampling and forward-inverse round trip error measurement |
| `src/quadruped_gait/algorithm/gait.py` | Duty factor and phase offset scheduler, gait library, contact schedule |
| `src/quadruped_gait/algorithm/swing.py` | Cycloidal and Bezier swing trajectories, de Casteljau evaluation, stance constraint |
| `src/quadruped_gait/algorithm/stability.py` | Convex hull support polygon, static and longitudinal stability margins |
| `src/quadruped_gait/pipeline/simulator.py` | Closed form trunk motion, foothold planning, and the recorded trace |
| `src/quadruped_gait/pipeline/sweep.py` | Batch runs over a parameter axis, used for the duty factor sweep |
| `src/quadruped_gait/pipeline/presets.py` | Named configurations shared by the examples and the regression test |
| `src/quadruped_gait/analysis/metrics.py` | Contact statistics, stability series and summaries, gait diagram intervals |
| `src/quadruped_gait/analysis/report.py` | Fixed width text rendering of every result table |
| `src/quadruped_gait/analysis/figures.py` | Gait diagram, stability history, foot path, and support polygon figures |
| `examples/` | Argument parsing and calls into the library, with no computation of their own |

The layers depend downward only. `model` imports nothing from the package,
`algorithm` imports `model`, `pipeline` imports both, and `analysis` imports all
three. Nothing in `model` or `algorithm` performs input or output, and no module
other than `analysis/figures.py` imports Matplotlib.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

There are 217 tests and the whole suite runs in about six seconds.

The property and invariant tier asserts that forward kinematics composed with
the inverse returns the original foot position, that unreachable targets are
reported for all three failure modes, that each gait is periodic with its
configured period, that the realised duty factor matches the command to within
one sample, that a trot has exactly two feet in stance, that a walk keeps at
least three, that the support polygon is empty below three non-collinear
contacts, that loaded feet neither slip nor leave the ground plane, and that the
sign convention of both margins matches values computed by hand on a unit square
and on a right triangle.

The regression tier recomputes the reference walk and compares the trunk pose,
foot positions, joint angles, contact flags, and both stability margins against
`tests/data/reference_walk.json` with a tolerance of 1e-9. Run
`uv run python tests/test_regression.py` to regenerate that file after a
reviewed change of behaviour.

The integration tier loads every script in `examples/`, runs it with a reduced
sample count, and checks that it exits cleanly and writes output. A separate
test asserts that no example is missing from that table, and one test exercises
the figure writing path.

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
  convention, and of the result that a quadruped creeping gait needs a duty
  factor of at least three quarters.
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
| [Matplotlib](https://matplotlib.org/) >= 3.9 | Gait diagram, stability history, foot path, and support polygon figures | Matplotlib licence, a BSD compatible PSF style licence |
| [pytest](https://pytest.org/) >= 8.3 | Test runner for all three test tiers | MIT |
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
