using BULA;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Body wrench in hull-local axes (x right, y up, z forward). Five components, not six: every
    /// gimballed thruster's force stays in a forward-up plane, so the rig has no lateral force
    /// authority at all and there is nothing to allocate along hull x.
    /// </summary>
    public struct GimbalWrench
    {
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
    /// </summary>
    [System.Serializable]
    public struct GimbalSettings
    {
        [Tooltip("Thrust of one thruster at full throttle, newtons.")]
        [Range(2000f, 40000f)] public float maxThrust;

        [Tooltip("Idle thrust, newtons. A thruster never shuts off, so its servo angle always matters.")]
        [Range(0f, 4000f)] public float minThrust;

        [Tooltip("Most backward-tilted servo angle, degrees. 0 is straight up.")]
        [Range(-85f, 0f)] public float servoMinDeg;

        [Tooltip("Most forward-tilted servo angle, degrees. 0 is straight up.")]
        [Range(0f, 85f)] public float servoMaxDeg;

        [Tooltip("Servo motor speed, degrees per second. Bounds this step's angle change.")]
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
    /// One step's view of the four-thruster rig. Component i of every float4 belongs to thruster i.
    ///
    /// Thruster i sits at hull-local (MountX, MountY, MountZ)[i], measured from the CENTER OF MASS,
    /// and produces force Health[i] · throttle[i] · MaxThrust · (0, cos θ, sin θ): the servo turns the
    /// nozzle about the hull's right axis, sweeping the exhaust forward → down → backward while the
    /// force sweeps backward → up → forward. θ = 0 is straight up.
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

        /// <summary>Absolute servo travel, radians.</summary>
        public float4 AngleLo, AngleHi;

        /// <summary>Absolute throttle range. A dead thruster is pinned to idle (Lo == Hi).</summary>
        public float4 ThrottleLo, ThrottleHi;

        /// <summary>This step's servo angle change bounds, radians.</summary>
        public float4 DAngleLo, DAngleHi;

        /// <summary>This step's throttle change bounds.</summary>
        public float4 DThrottleLo, DThrottleHi;

        /// <summary>Thrust at throttle 1, newtons.</summary>
        public float MaxThrust;

        /// <summary>Throttle the trim weight pulls toward — the even share across live thrusters.</summary>
        public float TrimThrottle;

        /// <summary>Newtons and newton-metres that count as a wrench residual of 1.</summary>
        public float ForceScale, TorqueScale;

        /// <summary>Quadratic costs on servo motion and on trim deviation.</summary>
        public float ServoWeight, TrimWeight;
    }

    /// <summary>
    /// Control allocation for four gimballed thrusters: choose 4 servo angles and 4 throttles whose
    /// combined wrench matches a demanded <see cref="GimbalWrench"/>, under the rig's angle range,
    /// angle rate, thrust range and thrust rate limits.
    ///
    /// Thrust turns with the servo as (cos θ, sin θ), so controls map to wrench NONLINEARLY and no
    /// fixed mixer matrix exists. Posed SQP-style instead: linearize the wrench about the CURRENT
    /// servo angles and throttles each step and solve for the change, which turns all four limit
    /// families into plain BOX constraints on the decision variables — the rate limits directly, the
    /// absolute ranges as the same box intersected with the distance left to each range. That is a
    /// convex box-constrained QP, solved by <c>QP.solve</c>.
    ///
    /// The servo rate limit is what makes the linearization honest: one step's angle change is at
    /// most rate·dt (a few degrees), well inside the range where cos/sin are linear.
    ///
    /// Controls are laid out as [θ0..θ3 radians, throttle0..throttle3 fractions]; the QP solves for
    /// the same layout of deltas. 8 controls against 5 reachable wrench components leave a
    /// 3-dimensional null space, which the servo and trim weights resolve.
    /// </summary>
    public static class GimbalAllocation
    {
        public const int Thrusters = 4;

        /// <summary>4 servo angles then 4 throttles.</summary>
        public const int ControlCount = 8;

        /// <summary>Wrench components the rig can reach: lift, drive, pitch, yaw, roll.</summary>
        public const int WrenchRows = 5;

        /// <summary>Servo angle of thruster i, radians.</summary>
        public static float Angle(in floatN z, int i) => z[i];

        /// <summary>Throttle of thruster i, fraction of <see cref="GimbalRig.MaxThrust"/>.</summary>
        public static float Throttle(in floatN z, int i) => z[Thrusters + i];

        /// <summary>Hull-local force direction of a servo at <paramref name="angle"/>: straight up at
        /// 0, forward at +90°, backward at -90°. Unit length.</summary>
        public static float3 ForceDirection(float angle)
        {
            math.sincos(angle, out float s, out float c);
            return new float3(0f, c, s);
        }

        /// <summary>
        /// Hull-frame wrench produced by controls <paramref name="z"/>, in newtons and newton-metres.
        /// Dead thrusters contribute nothing.
        /// </summary>
        public static GimbalWrench Wrench(in GimbalRig rig, in floatN z)
        {
            float4 mx = rig.MountX, my = rig.MountY, mz = rig.MountZ, health = rig.Health;

            var w = new GimbalWrench();
            for (int i = 0; i < Thrusters; i++)
            {
                math.sincos(z[i], out float s, out float c);
                float q = health[i] * z[Thrusters + i] * rig.MaxThrust;
                float fy = q * c, fz = q * s;
                float x = mx[i], y = my[i], zz = mz[i];

                w.Lift += fy;
                w.Drive += fz;
                w.Pitch += y * fz - zz * fy;
                w.Yaw += -x * fz;
                w.Roll += x * fy;
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
            float4 mx = rig.MountX, my = rig.MountY, mz = rig.MountZ, health = rig.Health;
            float invF = 1f / rig.ForceScale, invT = 1f / rig.TorqueScale;

            for (int i = 0; i < Thrusters; i++)
            {
                int ct = i, cu = Thrusters + i;
                math.sincos(z[ct], out float s, out float c);

                float x = mx[i], y = my[i], zz = mz[i];
                float g = health[i] * rig.MaxThrust;   // d|F| / d throttle
                float q = g * z[cu];                   // |F| now

                A[0, ct] = -q * s * invF;
                A[1, ct] = q * c * invF;
                A[2, ct] = q * (y * c + zz * s) * invT;
                A[3, ct] = -x * q * c * invT;
                A[4, ct] = -x * q * s * invT;

                A[0, cu] = g * c * invF;
                A[1, cu] = g * s * invF;
                A[2, cu] = g * (y * s - zz * c) * invT;
                A[3, cu] = -x * g * s * invT;
                A[4, cu] = x * g * c * invT;
            }
        }

        /// <summary>
        /// Builds the rig for one step: geometry and weights, the absolute ranges, and the delta box
        /// that <paramref name="z"/> and <paramref name="dt"/> leave. A dead thruster's throttle range
        /// collapses to idle, so it spools down at the power slew rate instead of snapping off.
        /// <paramref name="weight"/> is the hull's weight in newtons and <paramref name="arm"/> the
        /// lever arm that turns it into the torque scale.
        /// </summary>
        public static GimbalRig BuildRig(in GimbalSettings s,
            float4 mountX, float4 mountY, float4 mountZ, float4 health,
            in floatN z, float dt, float weight, float arm)
        {
            float maxThrust = math.max(s.maxThrust, 1f);
            float angLo = math.radians(math.min(s.servoMinDeg, s.servoMaxDeg));
            float angHi = math.radians(math.max(s.servoMinDeg, s.servoMaxDeg));
            float thrLo = math.saturate(math.min(s.minThrust, maxThrust) / maxThrust);
            float angRate = math.radians(s.servoRateDeg) * dt;
            float thrRate = (s.thrustRate / maxThrust) * dt;
            float live = math.max(math.csum(health), 1f);

            var rig = new GimbalRig
            {
                MountX = mountX,
                MountY = mountY,
                MountZ = mountZ,
                Health = health,
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

                DeltaBounds(z[i], angLo, angHi, angRate, out float daLo, out float daHi);
                rig.DAngleLo[i] = daLo;
                rig.DAngleHi[i] = daHi;

                DeltaBounds(z[Thrusters + i], thrLo, tHi, thrRate, out float dtLo, out float dtHi);
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
            e[0] = (w0.Lift - desired.Lift) / rig.ForceScale;
            e[1] = (w0.Drive - desired.Drive) / rig.ForceScale;
            e[2] = (w0.Pitch - desired.Pitch) / rig.TorqueScale;
            e[3] = (w0.Yaw - desired.Yaw) / rig.TorqueScale;
            e[4] = (w0.Roll - desired.Roll) / rig.TorqueScale;

            var Q = new floatMxN(ControlCount, ControlCount, Allocator.Temp, true);
            Blas.dotSym(in J, in J, ref Q);            // JᵀJ, exactly symmetric

            var c = new floatN(ControlCount, Allocator.Temp, true);
            Blas.dot(in e, in J, ref c);               // Jᵀe

            float4 tLo = rig.DThrottleLo, tHi = rig.DThrottleHi;
            float4 aLo = rig.DAngleLo, aHi = rig.DAngleHi;
            var xl = new floatN(ControlCount, Allocator.Temp, true);
            var xu = new floatN(ControlCount, Allocator.Temp, true);

            // JᵀJ has rank 5 at most, so the two regularizers are what make Q positive definite --
            // including when a dead thruster leaves its two J columns entirely zero.
            for (int i = 0; i < Thrusters; i++)
            {
                Q[i, i] += rig.ServoWeight;
                Q[Thrusters + i, Thrusters + i] += rig.TrimWeight;
                c[Thrusters + i] += rig.TrimWeight * (z[Thrusters + i] - rig.TrimThrottle);

                xl[i] = aLo[i]; xu[i] = aHi[i];
                xl[Thrusters + i] = tLo[i]; xu[Thrusters + i] = tHi[i];
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
                z[Thrusters + i] = math.clamp(z[Thrusters + i], tLo[i], tHi[i]);
            }
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
