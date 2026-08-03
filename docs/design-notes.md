# Design notes for Quadruped Gait Simulator

## Scope of the model

This is a kinematic and quasi-static model. It does not integrate rigid body
dynamics and it does not compute contact forces. Every statement it produces is
a statement about geometry: where the feet are, which of them are loaded, what
polygon they span, and how far the vertically projected centre of mass is from
the boundary of that polygon.

The final section, [What does not follow from this model](#what-does-not-follow-from-this-model),
lists the conclusions that are therefore out of reach. Read it before quoting
any number from this repository in a design decision.

## Method selection

### Leg kinematics: closed form inverse of a roll-pitch-pitch chain

The leg is a hip roll joint about the trunk x axis, a fixed lateral offset to
the hip pitch axis, then hip pitch and knee pitch about parallel y axes. This is
the layout of ANYmal (Hutter et al., 2016) and of the MIT Cheetah family (Bledt
et al., 2018), so results transfer to the machines the reader is likely to have
in mind.

The inverse is solved in closed form rather than numerically. After the roll
rotation is removed, the remainder is the planar two link problem of Siciliano
et al. (2009), section 2.12: the law of cosines gives the knee angle and one two
argument arctangent gives the hip pitch. The roll angle follows from the
difference of two arctangents in the lateral plane.

The assumptions this depends on, and where they hold:

- The two pitch axes are exactly parallel. This is true for every commercial
  quadruped leg of this class. If a manufacturing tolerance broke it, the closed
  form would be wrong by an amount proportional to the misalignment.
- The foot is a point. This is close enough for a small rubber ball foot on flat
  ground. It is not true for a large or a compliant foot, where the contact
  point moves under the foot as the leg rotates.
- The links are rigid. This is the standard assumption at these loads.

Two solution branches remain after the geometry is fixed. The knee sign is left
to the caller through `knee_forward`. The choice of whether the foot lies above
or below the hip pitch axis is fixed by convention: the solver always returns
the posture with the foot below the axis, since that is the only branch a
standing or walking robot uses. `foot_below_hip_axis` exposes the predicate so
that the convention is testable rather than implicit, and the docstring of
`inverse_kinematics` states that the function is a left inverse of the forward
map exactly on that branch.

Rejected: a numerical inverse by damped least squares. It would have handled an
arbitrary chain, including one with non-parallel axes, and would have degraded
gracefully near singularities. It was rejected because it introduces an
iteration count, a convergence tolerance, and a seed posture into what should be
a deterministic geometric fact, and because it cannot cleanly distinguish an
unreachable target from slow convergence. The closed form reports unreachability
as an exception carrying the measured distance and the limit, which is what a
planner needs.

### Gait definition: duty factor and relative phase

Gaits are parameterised as in Hildebrand (1965, 1989): a period, a duty factor,
and one relative phase per leg. The four library gaits use published footfall
patterns. The walk is the lateral sequence walk, footfall order hind-left,
front-left, hind-right, front-right at quarter cycle spacing, with a duty factor
of 0.75. The trot pairs diagonal legs, the pace pairs lateral legs, and the
bound pairs the fore legs against the hind legs, with a bound duty factor of 0.4
that produces two aerial phases per cycle.

The assumption is that the gait is symmetrical and periodic. That holds for
steady locomotion on flat ground. It does not hold for a gait transition, for a
free gait chosen foothold by foothold on rough terrain, or for a turn tight
enough that the inner and outer legs need different stride lengths.

One implementation detail is worth recording. A phase computed as
`(t / T - phi) mod 1` can land a fraction of a floating point epsilon below one
at an instant where it should be exactly zero, which flips a foot from stance to
swing at a support transition and briefly breaks the invariant that a walk keeps
three feet down. Phases within 1e-9 of a full cycle are therefore snapped to
zero. The tests avoid depending on how any individual boundary sample rounds, by
sampling at times that never coincide with a boundary.

Rejected: a central pattern generator built from coupled oscillators. It would
have produced gait transitions and adaptation for free, and it is what much of
the biological locomotion literature uses. It was rejected because the phase of
an oscillator network is an emergent quantity, so the commanded duty factor is
no longer a parameter that can be checked against a realised value. Since one of
the stated requirements is to verify that the realised contact fraction matches
the commanded duty factor, an explicit scheduler is the only design that makes
the check meaningful.

### Contact fractions: measured in closed form rather than counted

The stance indicator of one leg is a periodic step function of time, so the time
that leg spends loaded within a window is an integral, not a tally. Writing
`beta` for the duty factor, the measure of the stance set within `[0, x]` cycles
is `floor(x) beta + min(x - floor(x), beta)`, and the measure over any interval
is the difference of that expression at the two ends. The same formula holds for
negative arguments, which is what makes the difference valid over a window that
starts before the origin. `GaitParameters.stance_measure` is exactly this, and
`stance_intervals` enumerates the loaded intervals directly from the touchdown
instants `t = (k + phi_i) T`.

The consequence is that the realised duty factor a run reports carries no
discretisation error. It was previously the fraction of sampled instants at which
a leg was loaded, which is wrong by up to one sampling interval and which is why
a bound at 200 samples per cycle used to report a realised duty factor of 0.4042
against a commanded 0.4000.

The sampled measurement is kept beside the closed form one rather than deleted,
because it answers a different question. The closed form value is a property of
the gait parameters and would still be reported correctly if the simulator wrote
contact flags that contradicted its own scheduler. The counted value is a
measurement of what the trace actually holds. Reporting both, and asserting in
the tests that they agree to within one sampling interval, keeps the audit that
the closed form value on its own would have lost.

Rejected: refining the sampling rate until the counted value was accurate enough.
Halving the error costs twice the samples and twice the hull constructions in the
stability analysis, and it never reaches an exact answer. A closed form that is
correct at any sampling rate is both cheaper and stronger.

### Foot trajectories: neutral point footholds, cycloidal or Bezier swing

A foothold is placed at the neutral point: the ground point under the leg's
nominal hip position at the middle of the coming stance phase. This makes the
stance sweep symmetric about the hip, which is Raibert's symmetry condition for
steady locomotion (Raibert, 1986). Because the trunk motion is available in
closed form, the neutral point at a future instant is computed exactly rather
than extrapolated.

The swing curve offers two profiles. The cycloidal profile uses the horizontal
displacement `s - sin(2 pi s) / (2 pi)` and the raised cosine height
`(1 - cos(2 pi s)) / 2`, giving zero horizontal velocity at both ends and an
apex exactly at the requested clearance. The Bezier profile is a quintic with
the control fractions `(0, 0, 0.25, 0.75, 1, 1)` along the chord, so the first
and last control edges are vertical and the foot leaves and meets the ground
along the surface normal. Both reproduce their endpoints exactly and both are
monotone along the chord, which the tests assert.

Stance needs no curve at all. A loaded foot does not slip, so its world position
is constant and all of its motion in the trunk frame is the commanded body
twist. That is the whole stance model, and it is why the simulator has no
integration step.

Rejected: a swing trajectory optimised against a cost, for example minimum jerk
subject to a clearance constraint. It would have given smoother joint velocity
profiles. It was rejected because the joint velocity profile only matters once
actuator limits and dynamics are in the model, and neither is. Adding an
optimiser here would create the impression that the resulting trajectory is
dynamically feasible, which nothing in this repository checks.

### Stability: static and longitudinal margins on the convex hull

The support polygon is the convex hull of the loaded feet projected onto the
ground plane. The static stability margin is the shortest distance from the
vertically projected centre of mass to the polygon boundary, positive inside and
negative outside. The longitudinal stability margin is the same quantity
restricted to the direction of travel. Both definitions and the sign convention
are from McGhee and Frank (1968).

The hull comes from `scipy.spatial.ConvexHull`, which wraps Qhull (Barber et
al., 1996). Degenerate inputs are handled before and after the call: fewer than
three points, or any non-finite coordinate, short circuit to an empty polygon,
and a hull whose area falls below 1e-10 square metres is treated as collinear
and also returns empty. Both margins then return `nan`. This is deliberate. A
trot has two feet down, the hull is a line segment, and there is no margin to
report. Returning zero, or returning the distance to the segment, would both
invite the reader to compare a trot against a walk on a scale where the
comparison is meaningless.

The longitudinal margin is computed by clipping the infinite line through the
centre of mass against the inward half planes of the hull, which yields an
interval of signed distances `[t0, t1]`. The margin is `min(t1, -t0)`. Inside
the hull that is the smaller of the forward and backward distances to the
boundary and is positive. Outside it, but on a line that still crosses the hull,
it is the negative of the distance to the nearer crossing. If the line misses
the hull entirely it is `nan`.

The assumptions are: flat, level, rigid ground; a foot that neither slips nor
tips; friction sufficient that the constraint is purely geometric; and motion
slow enough that inertial terms are negligible against gravity. The last one is
the binding assumption, and it is exactly what fails for a trot.

Rejected: the zero moment point. It would have extended the criterion to faster
motion by accounting for the inertial wrench of the trunk. It was rejected
because the zero moment point needs a mass and an inertia tensor for every link
and the accelerations of every body, and this model deliberately carries
neither. Computing it from a kinematic model alone would require inventing an
inertia distribution, and any margin derived from invented numbers is worse than
no margin.

Rejected: the energy stability margin of Messuri and Klein (1985), which
measures the work needed to tip the machine over each support edge rather than
the horizontal distance to it. It is the better criterion on uneven ground and
for machines with a high centre of mass, because it accounts for the height the
centre of mass must rise. It was rejected because it needs the mass, which this
model does not carry, and because on the flat level ground assumed here it is a
monotone function of the static margin, so it would rank the same gaits in the
same order at the cost of an extra parameter.

Rejected: a full rigid body simulation with a contact solver. It would have
answered questions this model cannot, and it is the correct next step. It was
rejected for this repository because it would introduce a solver whose contact
model, integrator, and tuning would dominate the results, and because the stated
constraint was a self-contained implementation with no physics engine
dependency.

### Trunk motion: closed form rather than integrated

The commanded planar twist is constant in the trunk frame, so the trunk traces a
straight line when the yaw rate is zero and a circular arc otherwise. Both cases
have a closed form, which the simulator evaluates directly. The consequences are
that the trace has no accumulated integration error, that any sample can be
recomputed independently of the others, and that the foothold planner can
evaluate a future hip position exactly rather than predicting it.

The lateral trunk offset that improves the margin of a crawl gait is a sinusoid
of one cycle per gait period, added in the trunk frame. Displacing the trunk
toward the supporting side during each swing is the standard technique for
statically stable crawl gaits (Song and Waldron, 1989). Its limit is visible in
the results: at a duty factor of exactly 0.75 no offset produces a positive
minimum margin, because the required sign of the offset reverses at the very
instant the critical support edge passes under the centre of mass.

## Rejected alternatives

The method level alternatives are recorded in the section above, next to the
method they were weighed against. Three broader design choices are recorded
here.

### Storing the trace as an array of records rather than arrays of columns

The trace holds one array per quantity, each with a leading sample axis, and
exposes a `sample(i)` accessor that assembles a record on demand. The
alternative was a tuple of record objects.

Cost of the chosen design: a caller who wants one instant pays a small assembly
cost, and the class has more fields.

What it buys: the analysis layer works on whole columns, so the stability series
over 600 samples is a loop over hull constructions rather than a loop over
object attribute lookups, and the regression fixture is a direct slice of the
stored arrays. The record view is retained because it is the natural shape for
reasoning about a single instant.

### Fixing the leg count at four

Every array with a leg axis has exactly four entries and `LegId` has four
members. The alternative was a general n-legged machine.

Cost: hexapods and bipeds are out of scope, and generalising later would touch
the geometry, the gait library, and the contact state.

What it buys: the gait library can state published quadruped footfall patterns
directly, the invariants under test can be specific rather than conditional, and
the stability results can be compared against the quadruped-specific analysis of
McGhee and Frank. A general implementation would have had to state every
invariant in a form that holds for any leg count, which for the three quarter
duty factor threshold is simply not possible.

### Distributing the gait definitions in code rather than in a data file

The four gaits are module level constants behind a read-only mapping. The
alternative was a TOML or JSON gait catalogue.

Cost: adding a gait means editing Python.

What it buys: each gait carries its citation as a comment next to its phase
offsets, the values are covered by the type checker, and there is no file
loading path to test or to fail at runtime. `GaitParameters` is public, so a
caller who needs a gait outside the library constructs one directly.

### Choosing figure colours for colour vision deficiency rather than by default

The four leg colours were the Matplotlib default cycle. Under simulated
deuteranopia the green and the red of that cycle separate by about 4 units of
OKLab distance, which is below the threshold at which two marks can be told
apart, so a reader with the most common form of colour blindness could not
distinguish the hind-left leg from the front-right one.

Cost: the figures no longer match the default palette a reader may recognise
from other Matplotlib output, and the four hues had to be checked rather than
chosen.

What it buys: every pair of leg colours now separates by at least 8 units under
protanopia and deuteranopia. Colour is in any case never the only encoding: the
gait diagrams name each leg on the axis, the support polygon figure carries a
legend and distinguishes loaded from swinging feet by marker shape, and the duty
factor sweep separates certified from uncertified points by marker fill rather
than by hue.

## Known limitations

1. **Fast gaits.** The quasi-static criterion assumes inertial terms are
   negligible. It has no way to express this, so it silently degrades as speed
   rises. For a trot, a pace, or a bound it does not degrade, it simply returns
   `nan`, which is the honest outcome. For a fast walk it will still return a
   positive margin that no longer means the robot will stay upright. Removing
   this limitation requires a dynamic criterion and therefore a mass model.

2. **Non-level or non-rigid ground.** The ground is the plane `z = 0`
   everywhere. Footholds are projected onto it and the support polygon is built
   in it. On a slope, on a step, or on compliant ground the polygon and the
   projection direction are both wrong. Removing this requires a terrain map, a
   per-foothold height query, and a projection along gravity rather than along
   the world z axis.

3. **Unlimited friction.** Nothing checks that the required tangential force
   lies inside a friction cone, because no force is computed. A gait this model
   calls stable can still slip. Removing this requires a contact force
   distribution, which requires the dynamics.

4. **No joint or actuator limits.** The inverse kinematics rejects targets
   outside the geometric workspace, but nothing rejects a posture that a real
   joint could not reach, and nothing rejects a swing that would need more joint
   velocity or torque than an actuator can deliver. The default sampling limits
   in `model/workspace.py` are used only for the workspace study, not as
   constraints on the simulator.

5. **Point feet with no orientation.** The foot is a point, so the model cannot
   represent tipping about a foot edge, a rolling contact, or the ankle torque a
   larger foot would provide.

6. **Instantaneous contact transitions.** A foot is either loaded or not,
   switching at the phase boundary. There is no load transfer interval and no
   impact at touchdown, so the model cannot say anything about impact forces or
   about the momentum lost at each footfall.

7. **A single centre of mass fixed in the trunk.** The centre of mass is a fixed
   point in the trunk frame. In reality the legs carry a significant fraction of
   the mass and the whole body centre of mass moves as they swing. Removing this
   requires per-link masses, at which point the model is no longer purely
   kinematic.

8. **Sampled stability extrema.** The minimum static margin a run reports is the
   smallest value over the sampled instants, not the infimum over the interval.
   For the gaits studied here the margin is piecewise smooth and its minima fall
   at support transitions, which are sample times whenever the sampling rate
   divides the cycle evenly, so the reported value is right in practice. It is
   not guaranteed. Removing this needs the margin as a function of time in closed
   form, which needs the identity of the critical support edge as a function of
   time. That is tractable and is the obvious next thing to close.

## Closed limitations

### Sampled contact fractions, closed

**What it was.** The realised duty factor was measured by counting the samples at
which a leg was loaded, so it carried a discretisation error of up to one
sampling interval. At 200 samples per cycle that is 0.005, and the bound reported
a realised duty factor of 0.4042 against a commanded 0.4000. The corresponding
test could only assert a tolerance proportional to the sampling rate.

**What replaced it.** The closed form measure described under
[Contact fractions](#contact-fractions-measured-in-closed-form-rather-than-counted).
`ContactSummary` now carries `exact_duty_factors` and a `mean_stance_count`
computed from them, with `sampled_duty_factors` and `sampled_mean_stance_count`
beside them as the cross check. `contact_intervals` builds the gait diagram from
the schedule rather than by scanning the recorded flags, so a diagram edge lands
on the phase boundary rather than on the nearest sample. `stance_count_extrema`
returns the exact range of loaded feet over a cycle by evaluating the count once
between consecutive events, which is what lets the walk be described as never
dropping below three feet rather than as not having been observed to.

**What it cost.**

- Two more fields on `ContactSummary` and two more rows in the text report. A
  reader now has to be told which of two numbers is the answer, where before
  there was only one number and it was slightly wrong.
- A value derived from the gait parameters cannot audit the simulator. The
  sampled measurement is therefore retained, together with a test asserting that
  the two agree to within one sampling interval, which is a test that did not
  need to exist before.
- The gait diagram no longer depicts the recorded contact array directly. A test
  now asserts that every sample falls inside an exact interval exactly when its
  recorded flag is set, which restores that guarantee explicitly.

**What remains.** Only the duty factor and the loaded interval boundaries became
exact. The stance count histogram still counts samples, and so does the
supported fraction, and the minimum and mean stability margins are still extrema
and means over samples. That last one is now recorded as limitation 8 above,
since it was previously hidden inside this entry.

## What does not follow from this model

The following conclusions are not supported by anything in this repository, and
should not be drawn from its output:

- That a gait the model calls statically stable will keep the robot upright. The
  model checks a geometric condition on flat rigid ground with unlimited
  friction and no inertia. Falling is a dynamic event.
- That a gait the model reports as `n/a`, meaning the trot, the pace, and the
  bound, is unstable. The model has no criterion that applies to them. It is
  reporting that it cannot answer, not that the answer is negative.
- That the joint angles in a trace are achievable by a real actuator. No torque,
  no joint velocity, and no joint limit is checked.
- That the required friction is available. No tangential force is computed.
- That a larger static margin implies a larger margin against tipping on uneven
  ground. That is what the energy stability margin measures, and it is not
  computed here.
- That the swing trajectories are dynamically feasible or energetically
  sensible. They are geometric curves chosen for their endpoint and clearance
  properties alone.
- That the stability threshold of a duty factor of 0.75 transfers to a robot
  with a different hip layout or with a centre of mass away from the trunk
  origin. The threshold is a property of the symmetric machine analysed here.
  The mechanism, that the critical support edge is shared by the outgoing and
  the incoming support triangle, is general, but the numeric threshold is not.
