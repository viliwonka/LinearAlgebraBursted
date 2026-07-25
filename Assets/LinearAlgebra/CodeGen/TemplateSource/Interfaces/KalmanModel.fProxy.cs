namespace BULA
{
    /// <summary>
    /// Nonlinear state-transition model for the Extended Kalman Filter
    /// (<see cref="Kalman.ekfPredict{TModel}"/>), a Burst struct-functor matching
    /// <see cref="IfProxyLinearOperator"/>'s shape. The analytic Jacobian is REQUIRED (no numeric-
    /// differentiation fallback baked into the interface) -- call
    /// <see cref="Kalman.numericJacobianF{TModel}(in TModel, in fProxyN, in fProxyN, ref fProxyMxN)"/>
    /// from inside <see cref="JacobianF"/> for a model too awkward to differentiate by hand.
    /// Implement on a small, blittable struct.
    /// </summary>
    public interface IfProxyKFModel
    {
        /// <summary>xNext = f(x, u), the nonlinear one-step state propagation. xNext must be
        /// distinct from x.</summary>
        void F(in fProxyN x, in fProxyN u, ref fProxyN xNext);

        /// <summary>J = df/dx, evaluated at (x, u). J is n x n.</summary>
        void JacobianF(in fProxyN x, in fProxyN u, ref fProxyMxN J);
    }

    /// <summary>
    /// Nonlinear measurement model for the Extended Kalman Filter
    /// (<see cref="Kalman.ekfUpdate{TMeas}"/>). Same required-analytic-Jacobian contract as
    /// <see cref="IfProxyKFModel"/> -- use
    /// <see cref="Kalman.numericJacobianH{TMeas}(in TMeas, in fProxyN, ref fProxyMxN)"/> from inside
    /// <see cref="JacobianH"/> when needed.
    /// </summary>
    public interface IfProxyKFMeasurement
    {
        /// <summary>z = h(x), the nonlinear measurement function. z must be distinct from x.</summary>
        void H(in fProxyN x, ref fProxyN z);

        /// <summary>J = dh/dx, evaluated at x. J is m x n (m = the measurement dimension).</summary>
        void JacobianH(in fProxyN x, ref fProxyMxN J);
    }
}
