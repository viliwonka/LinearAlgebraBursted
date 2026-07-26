using BULA;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Body wrench in hull-local axes (x right, y up, z forward). Six components: the three FORCE
    /// rows first, then the three TORQUE rows, each triple ordered x, y, z. That is also the row
    /// order of <see cref="GimbalAllocation.ScaledJacobian"/> and of the residual
    /// <see cref="GimbalAllocation.Solve"/> builds, so the three must be edited together.
    ///
    /// Torques are r x F about the center of mass. Force rows enter the QP divided by
    /// <see cref="GimbalRig.ForceScale"/>, torque rows by <see cref="GimbalRig.TorqueScale"/>.
    /// </summary>
    public struct GimbalWrench
    {
        /// <summary>Force along hull +x, newtons. Positive pushes the hull right — the strafe axis.</summary>
        public float Lateral;

        /// <summary>Force along hull +y, newtons.</summary>
        public float Lift;

        /// <summary>Force along hull +z, newtons.</summary>
        public float Drive;

        /// <summary>Torque about hull +x, newton-metres. Positive pitches the nose down.</summary>
        public float Pitch;

        /// <summary>Torque about hull +y, newton-metres. Positive yaws the nose right.</summary>
        public float Yaw;

        /// <summary>Torque about hull +z, newton-metres. Positive rolls the right side up.</summary>
        public float Roll;
    }

    /// <summary>
    /// Actuator limits and allocation weights shared by all four thrusters — the inspector face of
    /// the rig. The four limit families (angle range, angle rate, thrust range, thrust rate) are the
    /// constraints the allocation QP is solved under; tightening either RATE is what makes the tank
    /// sluggish, since a rate is a bound on this step's change, not on the value.
    ///
    /// One angle range and one angle rate cover BOTH gimbal axes of every nozzle.
    /// </summary>
    [System.Serializable]
    public struct GimbalSettings
    {
        [Tooltip("Thrust of one thruster at full throttle, newtons.")]
        [Range(2000f, 40000f)] public float maxThrust;

        [Tooltip("Idle thrust, newtons. A thruster never shuts off, so its gimbal angles always matter.")]
        [Range(0f, 4000f)] public float minThrust;

        [Tooltip("Most negative gimbal angle on either axis, degrees. 0 is straight up.")]
        [Range(-GimbalAllocation.MaxGimbalDeg, 0f)] public float servoMinDeg;

        [Tooltip("Most positive gimbal angle on either axis, degrees. 0 is straight up.")]
        [Range(0f, GimbalAllocation.MaxGimbalDeg)] public float servoMaxDeg;

        [Tooltip("Servo motor speed, degrees per second. Bounds this step's angle change on either axis.")]
        [Range(15f, 720f)] public float servoRateDeg;

        [Tooltip("Power slew rate, newtons per second. Bounds this step's thrust change.")]
        [Range(2000f, 400000f)] public float thrustRate;

        [Tooltip("Cost on servo motion. Picks a point in the allocation's null space.")]
        [Range(0.001f, 2f)] public float servoWeight;

        [Tooltip("Cost on leaving the even share of the hull's weight. Also picks the null space.")]
        [Range(0.001f, 2f)] public float trimWeight;

        public static GimbalSettings Default => new GimbalSettings
        {
            maxThrust = 12000f,
            minThrust = 300f,
            servoMinDeg = -55f,
            servoMaxDeg = 55f,
            servoRateDeg = 200f,
            thrustRate = 60000f,
            servoWeight = 0.05f,
            trimWeight = 0.08f,
        };
    }

    /// <summary>
    /// Cheeseman-Bennett rotor ground effect: a nozzle close to the ground pushes against its own
    /// reflected downwash and delivers MORE thrust than was commanded,
    ///
    ///     T_effective = T_commanded / (1 - (R / 4z)²)
    ///
    /// for effective nozzle radius R and nozzle height above ground z. Evaluated PER NOZZLE, so a
    /// tilted hull is augmented differently at each corner.
    /// </summary>
    public static class GroundEffect
    {
        /// <summary>
        /// Largest augmentation <see cref="Factor(float, float)"/> may report.
        ///
        /// The correlation is a fit to measured rotor data down to roughly z = R/2, where it reads
        /// 1.33; below that it is extrapolation running into a pole at z = R/4, and it is negative
        /// underneath. 1.5 is reached at z = 0.433·R, just under where the curve stops being data, so
        /// the clamp takes over exactly where the model does. It is also the bound on how far the rest
        /// of the loop is asked to move in one operating point: lift authority, the hover model's
        /// vertical input column and the throttle needed to carry the hull all change by at most half.
        /// </summary>
        public const float MaxFactor = 1.5f;

        /// <summary>
        /// Nozzle height at which the model reaches <see cref="MaxFactor"/>, metres. The factor is
        /// held there for anything lower.
        /// </summary>
        public static float ClampHeight(float radius) => radius / (4f * math.sqrt(1f - 1f / MaxFactor));

        /// <summary>
        /// Thrust multiplier for a nozzle of effective radius <paramref name="radius"/> whose exit
        /// plane is <paramref name="height"/> metres above the ground. Always finite and in
        /// [1, <see cref="MaxFactor"/>]: a radius of zero or less disables the effect and returns 1,
        /// and any height at or below <see cref="ClampHeight"/> — including a nozzle at, or under, the
        /// surface — returns <see cref="MaxFactor"/>, so the pole is never evaluated.
        /// </summary>
        public static float Factor(float height, float radius)
        {
            if (radius <= 0f) return 1f;
            float t = radius / (4f * math.max(height, ClampHeight(radius)));
            // The height clamp is what keeps the pole out of reach; the min only holds the stated
            // range exactly against the rounding of ClampHeight's own square root.
            return math.min(1f / (1f - t * t), MaxFactor);
        }

        /// <summary>Per-nozzle <see cref="Factor(float, float)"/> for all four heights at once.</summary>
        public static float4 Factor(float4 height, float radius)
        {
            if (radius <= 0f) return new float4(1f);
            float4 t = radius / (4f * math.max(height, new float4(ClampHeight(radius))));
            return math.min(1f / (1f - t * t), new float4(MaxFactor));
        }
    }

    /// <summary>
    /// One step's view of the four-thruster rig. Component i of every float4 belongs to thruster i.
    ///
    /// Thruster i sits at hull-local (MountX, MountY, MountZ)[i], measured from the CENTER OF MASS,
    /// and produces force Health[i] · GroundGain[i] · throttle[i] · MaxThrust ·
    /// <see cref="GimbalAllocation.ForceDirection"/>(pitch[i], yaw[i]). Both angles are zero when the
    /// nozzle points straight down and the thrust straight up.
    ///
    /// Throttle is a fraction of <see cref="MaxThrust"/>, so both control families are O(1) and the
    /// allocation Hessian is not split across a newtons-versus-radians scale gap.
    ///
    /// The D* delta boxes are the rate limits intersected with the absolute ranges AROUND THE
    /// CONTROLS THE RIG WAS BUILT FROM, so a rig is valid for one step and one control vector only
    /// (<see cref="GimbalAllocation.BuildRig"/>).
    /// </summary>
    public struct GimbalRig
    {
        /// <summary>Hull-local mount positions relative to the center of mass, metres.</summary>
        public float4 MountX, MountY, MountZ;

        /// <summary>Per-thruster effectiveness in [0, 1]. 0 is a dead thruster.</summary>
        public float4 Health;

        /// <summary>
        /// Per-thruster thrust multiplier at this step's operating point — the
        /// <see cref="GroundEffect.Factor(float, float)"/> of that nozzle's own height above the
        /// ground. 1 is out of ground effect. This is what the allocation sizes throttles against, so
        /// it must be the SAME number the applied force is scaled by.
        /// </summary>
        public float4 GroundGain;

        /// <summary>Absolute gimbal travel, radians. Shared by both axes of every nozzle.</summary>
        public float4 AngleLo, AngleHi;

        /// <summary>Absolute throttle range. A dead thruster is pinned to idle (Lo == Hi).</summary>
        public float4 ThrottleLo, ThrottleHi;

        /// <summary>This step's pitch-axis angle change bounds, radians.</summary>
        public float4 DPitchLo, DPitchHi;

        /// <summary>This step's yaw-axis angle change bounds, radians.</summary>
        public float4 DYawLo, DYawHi;

        /// <summary>This step's throttle change bounds.</summary>
        public float4 DThrottleLo, DThrottleHi;

        /// <summary>Thrust at throttle 1, newtons.</summary>
        public float MaxThrust;

        /// <summary>
        /// Throttle the trim weight pulls toward — the even share across live thrusters, weighted by
        /// what each of them actually delivers (<see cref="GroundGain"/>).
        /// </summary>
        public float TrimThrottle;

        /// <summary>Newtons and newton-metres that count as a wrench residual of 1.</summary>
        public float ForceScale, TorqueScale;

        /// <summary>Quadratic costs on servo motion and on trim deviation.</summary>
        public float ServoWeight, TrimWeight;
    }

    /// <summary>
    /// Control allocation for four TWO-AXIS gimballed thrusters: choose 4 pitch angles, 4 yaw angles
    /// and 4 throttles whose combined wrench matches a demanded <see cref="GimbalWrench"/>, under the
    /// rig's angle range, angle rate, thrust range and thrust rate limits.
    ///
    /// Each nozzle steers over a spherical cap rather than an arc, so lateral force is reachable and
    /// all six wrench components can be commanded independently.
    ///
    /// Thrust turns with the servos as sin/cos, so controls map to wrench NONLINEARLY and no fixed
    /// mixer matrix exists. Posed SQP-style instead: linearize the wrench about the CURRENT angles
    /// and throttles each step and solve for the change, which turns all four limit families into
    /// plain BOX constraints on the decision variables — the rate limits directly, the absolute
    /// ranges as the same box intersected with the distance left to each range. That is a convex
    /// box-constrained QP, solved by <c>QP.solve</c>.
    ///
    /// The servo rate limit is what makes the linearization honest: one step's angle change is at
    /// most rate·dt (a few degrees), well inside the range where cos/sin are linear.
    ///
    /// Controls are laid out as [pitch0..pitch3 radians, yaw0..yaw3 radians, throttle0..throttle3
    /// fractions]; the QP solves for the same layout of deltas. 12 controls against 6 wrench
    /// components leave a 6-dimensional null space, which the servo and trim weights resolve.
    /// </summary>
    public static class GimbalAllocation
    {
        public const int Thrusters = 4;

        /// <summary>4 pitch angles, then 4 yaw angles, then 4 throttles.</summary>
        public const int ControlCount = 12;

        /// <summary>Wrench components the rig can reach: lateral, lift, drive, pitch, yaw, roll.</summary>
        public const int WrenchRows = 6;

        /// <summary>
        /// Hard cap on either gimbal angle, degrees; <see cref="BuildRig"/> clamps the settings to it.
        ///
        /// This is a LIFT FLOOR, not a numerical guard. At pitch = ±90° the thrust lies flat along the
        /// hull's fore-aft axis and that nozzle carries none of the hull's weight, which is not a place
        /// a hover vehicle ever wants to be. The cap keeps cos(pitch) >= 0.5, so every nozzle always
        /// puts at least half its thrust into lift.
        ///
        /// Staying clear of the parametrization's degenerate direction (see
        /// <see cref="ForceDirection"/>) is a welcome side effect, not the reason: the regularized
        /// Hessian (<see cref="Hessian"/>) turns a rank drop into a min-norm step rather than an
        /// undefined one, so the allocation stays well posed at the pole regardless.
        /// </summary>
        public const float MaxGimbalDeg = 60f;

        /// <summary>Pitch-axis gimbal angle of thruster i, radians.</summary>
        public static float PitchAngle(in floatN z, int i) => z[i];

        /// <summary>Yaw-axis gimbal angle of thruster i, radians.</summary>
        public static float YawAngle(in floatN z, int i) => z[Thrusters + i];

        /// <summary>Throttle of thruster i, fraction of <see cref="GimbalRig.MaxThrust"/>.</summary>
        public static float Throttle(in floatN z, int i) => z[2 * Thrusters + i];

        /// <summary>
        /// Hull-local force direction of a nozzle gimballed to (<paramref name="pitch"/>,
        /// <paramref name="yaw"/>), radians. Straight up at (0, 0); pitch tilts the thrust toward
        /// hull +z at +90°, yaw tilts it toward hull +x at +90°. Unit length.
        ///
        /// WHERE THE POLE SITS is a property of this parametrization, and it was chosen. d(dir)/d(yaw)
        /// carries a cos(pitch) factor, so the yaw axis degenerates at pitch = ±90° — thrust fully fore
        /// or aft, a pose the vehicle is never flown to. Parametrize the same spherical cap as tilt
        /// magnitude plus azimuth instead and the degenerate direction moves to zero tilt, thrust
        /// straight up, which is exactly where a hover vehicle lives. Same cap, same algebra, and the
        /// singular pose goes from unreachable to permanent.
        /// </summary>
        public static float3 ForceDirection(float pitch, float yaw)
        {
            math.sincos(pitch, out float sp, out float cp);
            math.sincos(yaw, out float sy, out float cy);
            return new float3(sy * cp, cy * cp, sp);
        }

        /// <summary>
        /// Hull-frame wrench produced by controls <paramref name="z"/>, in newtons and newton-metres.
        /// Dead thrusters contribute nothing.
        /// </summary>
        public static GimbalWrench Wrench(in GimbalRig rig, in floatN z)
        {
            float4 mx = rig.MountX, my = rig.MountY, mz = rig.MountZ;
            float4 health = rig.Health, gain = rig.GroundGain;

            var w = new GimbalWrench();
            for (int i = 0; i < Thrusters; i++)
            {
                float q = health[i] * gain[i] * z[2 * Thrusters + i] * rig.MaxThrust;
                float3 f = q * ForceDirection(z[i], z[Thrusters + i]);
                float3 t = math.cross(new float3(mx[i], my[i], mz[i]), f);

                w.Lateral += f.x; w.Lift += f.y; w.Drive += f.z;
                w.Pitch += t.x; w.Yaw += t.y; w.Roll += t.z;
            }
            return w;
        }

        /// <summary>
        /// d(wrench)/d(controls) at <paramref name="z"/>, rows already divided by the rig's force and
        /// torque scales so every entry is dimensionless. <paramref name="A"/> is
        /// <see cref="WrenchRows"/> x <see cref="ControlCount"/> and is fully overwritten.
        /// </summary>
        public static void ScaledJacobian(in GimbalRig rig, in floatN z, ref floatMxN A)
        {
            float4 mx = rig.MountX, my = rig.MountY, mz = rig.MountZ;
            float4 health = rig.Health, gain = rig.GroundGain;
            float invF = 1f / rig.ForceScale, invT = 1f / rig.TorqueScale;

            for (int i = 0; i < Thrusters; i++)
            {
                int colP = i, colY = Thrusters + i, colU = 2 * Thrusters + i;
                math.sincos(z[colP], out float sp, out float cp);
                math.sincos(z[colY], out float sy, out float cy);

                float3 r = new float3(mx[i], my[i], mz[i]);
                float g = health[i] * gain[i] * rig.MaxThrust;   // d|F| / d throttle
                float q = g * z[colU];                 // |F| now

                // Direction derivatives of (sy·cp, cy·cp, sp). The yaw column's cos(pitch) factor is
                // where this parametrization degenerates; see ForceDirection for why that pose is
                // outside the envelope, and Hessian for what happens if it is entered anyway.
                Column(ref A, colP, q * new float3(-sy * sp, -cy * sp, cp), r, invF, invT);
                Column(ref A, colY, q * new float3(cy * cp, -sy * cp, 0f), r, invF, invT);
                Column(ref A, colU, g * new float3(sy * cp, cy * cp, sp), r, invF, invT);
            }
        }

        /// <summary>
        /// Q = JᵀJ plus the regularizer diagonals: <see cref="GimbalRig.ServoWeight"/> on all 8 angle
        /// controls and <see cref="GimbalRig.TrimWeight"/> on the 4 throttles. <paramref name="Q"/> is
        /// <see cref="ControlCount"/> square and is fully overwritten.
        ///
        /// JᵀJ has rank at most <see cref="WrenchRows"/> of <see cref="ControlCount"/>; a dead thruster
        /// zeroes three more columns and a nozzle at the parametrization's pole zeroes its yaw column.
        /// The two weights are the ONLY thing making Q positive definite, and they do it
        /// unconditionally — min eig(Q) >= min(ServoWeight, TrimWeight), which the settings keep above
        /// zero. Tikhonov, so a rank drop costs a min-norm step, not a failed solve.
        /// </summary>
        public static void Hessian(in GimbalRig rig, in floatMxN J, ref floatMxN Q)
        {
            Blas.dotSym(in J, in J, ref Q);            // JᵀJ, exactly symmetric

            for (int i = 0; i < Thrusters; i++)
            {
                Q[i, i] += rig.ServoWeight;
                Q[Thrusters + i, Thrusters + i] += rig.ServoWeight;
                Q[2 * Thrusters + i, 2 * Thrusters + i] += rig.TrimWeight;
            }
        }

        /// <summary>
        /// Builds the rig for one step: geometry and weights, the absolute ranges, and the delta box
        /// that <paramref name="z"/> and <paramref name="dt"/> leave. A dead thruster's throttle range
        /// collapses to idle, so it spools down at the power slew rate instead of snapping off.
        /// <paramref name="weight"/> is the hull's weight in newtons and <paramref name="arm"/> the
        /// lever arm that turns it into the torque scale. <paramref name="groundGain"/> is
        /// <see cref="GimbalRig.GroundGain"/>; pass 1 for a rig out of ground effect.
        /// </summary>
        public static GimbalRig BuildRig(in GimbalSettings s,
            float4 mountX, float4 mountY, float4 mountZ, float4 health, float4 groundGain,
            in floatN z, float dt, float weight, float arm)
        {
            float maxThrust = math.max(s.maxThrust, 1f);
            float lim = math.radians(MaxGimbalDeg);
            float angLo = math.clamp(math.radians(math.min(s.servoMinDeg, s.servoMaxDeg)), -lim, lim);
            float angHi = math.clamp(math.radians(math.max(s.servoMinDeg, s.servoMaxDeg)), -lim, lim);
            float thrLo = math.saturate(math.min(s.minThrust, maxThrust) / maxThrust);
            float angRate = math.radians(s.servoRateDeg) * dt;
            float thrRate = (s.thrustRate / maxThrust) * dt;
            // Lift the whole rig can raise per unit throttle, in units of one nominal thruster: what
            // the trim target has to be measured against, so the regularizer does not pull the
            // throttles back toward an out-of-ground-effect share the rig no longer needs.
            float live = math.max(math.csum(health * groundGain), 1f);

            var rig = new GimbalRig
            {
                MountX = mountX,
                MountY = mountY,
                MountZ = mountZ,
                Health = health,
                GroundGain = groundGain,
                MaxThrust = maxThrust,
                ForceScale = weight,
                TorqueScale = weight * arm,
                TrimThrottle = math.clamp(weight / (live * maxThrust), thrLo, 1f),
                ServoWeight = s.servoWeight,
                TrimWeight = s.trimWeight,
            };

            for (int i = 0; i < Thrusters; i++)
            {
                bool dead = health[i] <= 0f;
                float tHi = dead ? thrLo : 1f;

                rig.AngleLo[i] = angLo;
                rig.AngleHi[i] = angHi;
                rig.ThrottleLo[i] = thrLo;
                rig.ThrottleHi[i] = tHi;

                DeltaBounds(z[i], angLo, angHi, angRate, out float dpLo, out float dpHi);
                rig.DPitchLo[i] = dpLo;
                rig.DPitchHi[i] = dpHi;

                DeltaBounds(z[Thrusters + i], angLo, angHi, angRate, out float dyLo, out float dyHi);
                rig.DYawLo[i] = dyLo;
                rig.DYawHi[i] = dyHi;

                DeltaBounds(z[2 * Thrusters + i], thrLo, tHi, thrRate, out float dtLo, out float dtHi);
                rig.DThrottleLo[i] = dtLo;
                rig.DThrottleHi[i] = dtHi;
            }

            return rig;
        }

        /// <summary>
        /// Solves this step's allocation QP and applies the step: <paramref name="z"/> carries the
        /// current controls in and the commanded ones out. Nothing is applied unless the QP reaches
        /// an optimum, so a failed solve holds the last command rather than lurching.
        ///
        /// Job-safe: every buffer is Allocator.Temp and disposed before returning.
        /// <paramref name="maxIter"/> is the active-set pivot budget; 0 picks the library default.
        /// </summary>
        public static QPInfo Solve(in GimbalRig rig, in GimbalWrench desired, ref floatN z, int maxIter)
        {
            var J = new floatMxN(WrenchRows, ControlCount, Allocator.Temp, true);
            ScaledJacobian(in rig, in z, ref J);

            // Wrench error at the current controls, in the same scaled units as J: the QP minimizes
            // ½‖J·delta + e‖² plus the two regularizers.
            GimbalWrench w0 = Wrench(in rig, in z);
            var e = new floatN(WrenchRows, Allocator.Temp, true);
            e[0] = (w0.Lateral - desired.Lateral) / rig.ForceScale;
            e[1] = (w0.Lift - desired.Lift) / rig.ForceScale;
            e[2] = (w0.Drive - desired.Drive) / rig.ForceScale;
            e[3] = (w0.Pitch - desired.Pitch) / rig.TorqueScale;
            e[4] = (w0.Yaw - desired.Yaw) / rig.TorqueScale;
            e[5] = (w0.Roll - desired.Roll) / rig.TorqueScale;

            var Q = new floatMxN(ControlCount, ControlCount, Allocator.Temp, true);
            Hessian(in rig, in J, ref Q);

            var c = new floatN(ControlCount, Allocator.Temp, true);
            Blas.dot(in e, in J, ref c);               // Jᵀe

            float4 pLo = rig.DPitchLo, pHi = rig.DPitchHi;
            float4 yLo = rig.DYawLo, yHi = rig.DYawHi;
            float4 tLo = rig.DThrottleLo, tHi = rig.DThrottleHi;
            var xl = new floatN(ControlCount, Allocator.Temp, true);
            var xu = new floatN(ControlCount, Allocator.Temp, true);

            for (int i = 0; i < Thrusters; i++)
            {
                c[2 * Thrusters + i] += rig.TrimWeight * (z[2 * Thrusters + i] - rig.TrimThrottle);

                xl[i] = pLo[i]; xu[i] = pHi[i];
                xl[Thrusters + i] = yLo[i]; xu[Thrusters + i] = yHi[i];
                xl[2 * Thrusters + i] = tLo[i]; xu[2 * Thrusters + i] = tHi[i];
            }

            var A = new floatMxN(0, ControlCount, Allocator.Temp);
            var b = new floatN(0, Allocator.Temp);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var delta = new floatN(ControlCount, Allocator.Temp, true);

            QPInfo info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu,
                                   ref delta, out double _, maxIter);

            if (info)
            {
                for (int j = 0; j < ControlCount; j++) z[j] += delta[j];
                ClampToRanges(in rig, ref z);
            }

            delta.Dispose(); senses.Dispose(); b.Dispose(); A.Dispose();
            xu.Dispose(); xl.Dispose(); c.Dispose(); Q.Dispose(); e.Dispose(); J.Dispose();
            return info;
        }

        /// <summary>Clamps every control into the rig's absolute ranges, in place.</summary>
        public static void ClampToRanges(in GimbalRig rig, ref floatN z)
        {
            float4 aLo = rig.AngleLo, aHi = rig.AngleHi, tLo = rig.ThrottleLo, tHi = rig.ThrottleHi;
            for (int i = 0; i < Thrusters; i++)
            {
                z[i] = math.clamp(z[i], aLo[i], aHi[i]);
                z[Thrusters + i] = math.clamp(z[Thrusters + i], aLo[i], aHi[i]);
                z[2 * Thrusters + i] = math.clamp(z[2 * Thrusters + i], tLo[i], tHi[i]);
            }
        }

        // One Jacobian column: the force-derivative f already carries whatever control the column
        // differentiates, and the torque rows are the same r x F the wrench accumulates, so the row
        // order can only drift if both are edited.
        static void Column(ref floatMxN A, int col, float3 f, float3 r, float invF, float invT)
        {
            float3 t = math.cross(r, f);
            A[0, col] = f.x * invF;
            A[1, col] = f.y * invF;
            A[2, col] = f.z * invF;
            A[3, col] = t.x * invT;
            A[4, col] = t.y * invT;
            A[5, col] = t.z * invT;
        }

        // Feasible change box for one control: the actuator's rate window intersected with what is
        // left of its absolute range. The two are disjoint only when the state already sits outside a
        // range that moved (a limit dragged at runtime); the step is then the rate-feasible one that
        // gets closest to the range, so the actuator slews back instead of snapping. lo <= hi always,
        // which QP.solve requires of its bounds.
        static void DeltaBounds(float cur, float absLo, float absHi, float rate, out float lo, out float hi)
        {
            float travelLo = absLo - cur, travelHi = absHi - cur;
            lo = math.max(-rate, travelLo);
            hi = math.min(rate, travelHi);
            if (lo > hi)
                lo = hi = math.clamp(math.clamp(0f, travelLo, travelHi), -rate, rate);
        }
    }
}
