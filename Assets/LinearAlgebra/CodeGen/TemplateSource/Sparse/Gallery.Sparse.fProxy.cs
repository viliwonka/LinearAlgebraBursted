using System;
using Unity.Mathematics;
using Unity.Collections;
using BULA;
using BULA.Sparse;

namespace BULA.Gallery
{
    /// <summary>
    /// Sparse (BSR) test-matrix generators — the block-sparse counterpart to the dense gallery.
    /// Unlike the dense generators these return a <see cref="fProxyBSR"/> and, crucially, never
    /// materialize a dense form, so they are usable at N ≈ 10⁴ and beyond at low fill (a dense
    /// 10000×10000 matrix is ~800 MB in float; the sparse encoding stores only the nonzero blocks).
    /// Opt in with <c>using BULA.Gallery;</c>.
    ///
    /// Fill is at the BLOCK level: <c>density</c> is the target fraction of the blockRows×blockCols
    /// block grid that is nonzero (the diagonal blocks are always present). Actual DOF size is
    /// blockRows·BR. Determinism: fully seeded (<see cref="Unity.Mathematics.Random"/>), so a given
    /// (dims, density, seed) reproduces bit-for-bit.
    /// </summary>
    public static partial class fProxyGallery
    {
        // Off-diagonal block entries are drawn from [-OffScale, OffScale]; diagonal blocks are made
        // strongly enough dominant (per-row, exactly) that the matrix is guaranteed SPD / invertible
        // regardless of the random draw. See fProxyRandomSparseSPD.
        /*+choose[const float fProxySparseOffScale = 0.3f;|const double fProxySparseOffScale = 0.3;]*/const float fProxySparseOffScale = 0.3f;/*-choose*/

        /// <summary>
        /// Random symmetric positive-definite block-sparse matrix (blockRows·BR square). Each diagonal
        /// block is MᵀM + ρ·I with ρ chosen per block-row so the assembled matrix is STRICTLY block-row
        /// diagonally dominant — hence SPD by Gershgorin — for any random draw; off-diagonal blocks are
        /// added in symmetric (i,j)+(j,i) pairs at ~<paramref name="density"/> block fill. O(nnz) time
        /// and memory (no dense form). Use for CG/PCG/MINRES and the sparse eigensolvers.
        /// </summary>
        /// <param name="blockRows">Number of block rows/cols (square). DOF size = blockRows·BR.</param>
        /// <param name="BR">Square block size (2–6 hit the unrolled spMV kernels).</param>
        /// <param name="density">Target fraction of the block grid that is nonzero (e.g. 0.01–0.02).</param>
        /// <param name="seed">RNG seed (0 is remapped to 1; Unity.Mathematics.Random needs nonzero).</param>
        public static fProxyBSR fProxyRandomSparseSPD(int blockRows, int BR, fProxy density, uint seed, Allocator allocator = Allocator.Temp)
        {
            if (blockRows < 1) throw new ArgumentException("fProxyRandomSparseSPD: blockRows must be >= 1");
            if (BR < 1) throw new ArgumentException("fProxyRandomSparseSPD: BR must be >= 1");

            var rng = new Unity.Mathematics.Random(seed == 0 ? 1u : seed);
            int deg = (int)math.round((double)density * blockRows) - 1;
            if (deg < 0) deg = 0;
            if (deg > blockRows - 1) deg = blockRows - 1;

            var offBound = new fProxyN(blockRows, Allocator.Temp);
            for (int i = 0; i < blockRows; i++) offBound[i] = (fProxy)0;

            int nnzbEstimate = blockRows + blockRows * deg * 2;
            var builder = new fProxyBSRBuilder(blockRows, blockRows, BR, BR, Allocator.Temp, nnzbEstimate);

            var blk  = new fProxyMxN(BR, BR, Allocator.Temp);
            var blkT = new fProxyMxN(BR, BR, Allocator.Temp);
            fProxy blockRowBound = (fProxy)fProxySparseOffScale * BR;

            for (int i = 0; i < blockRows; i++)
            {
                for (int d = 0; d < deg; d++)
                {
                    int j = rng.NextInt(0, blockRows);
                    if (j == i) continue;
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                        {
                            fProxy v = (fProxy)/*+choose[rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale)|rng.NextDouble(-fProxySparseOffScale, fProxySparseOffScale)]*/rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale)/*-choose*/;
                            blk[r, c]  = v;
                            blkT[c, r] = v;
                        }
                    builder.AddBlock(i, j, in blk);
                    builder.AddBlock(j, i, in blkT);
                    offBound[i] += blockRowBound;
                    offBound[j] += blockRowBound;
                }
            }

            // Uses the ref-dest primitive with Di preallocated once outside the loop, instead of the
            // allocating convenience overload Blas.dot(Mi, Mi, true) (which would allocate a fresh
            // matrix every iteration). Identical kernel (matAtA), identical arithmetic.
            var Mi = new fProxyMxN(BR, BR, Allocator.Temp);
            var Di = new fProxyMxN(BR, BR, Allocator.Temp, true);
            for (int i = 0; i < blockRows; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = (fProxy)rng.NextFloat(-1f, 1f);

                Blas.dot(in Mi, in Mi, ref Di, true);     // Di = MᵀM (SPD, symmetric)
                fProxy rho = (fProxy)(BR * BR) + offBound[i] + (fProxy)1;
                for (int d = 0; d < BR; d++) Di[d, d] += rho;
                builder.AddBlock(i, i, in Di);
            }

            var result = builder.ToBSR(allocator);

            Di.Dispose();
            Mi.Dispose();
            blkT.Dispose();
            blk.Dispose();
            builder.Dispose();
            offBound.Dispose();

            return result;
        }

        /// <summary>
        /// Random block-sparse matrix that is NOT symmetric. Square (blockRows == blockCols) gives a
        /// strictly row-diagonally-dominant, hence invertible, non-symmetric matrix (use for BiCGSTAB /
        /// GMRES-style solvers). Tall (blockRows &gt; blockCols) gives a full-column-rank rectangular
        /// operator (the top blockCols square carries the dominant diagonal), suitable for the iterative
        /// least-squares solvers (lsqr/lsmr over a sparse operator). O(nnz), no dense form.
        /// </summary>
        /// <param name="blockRows">Block rows (m/BR).</param>
        /// <param name="blockCols">Block cols (n/BC). Equal → square; greater → tall rectangular.</param>
        /// <param name="BR">Square block size (BC == BR).</param>
        /// <param name="density">Target fraction of the block grid that is nonzero.</param>
        /// <param name="seed">RNG seed (0 → 1).</param>
        public static fProxyBSR fProxyRandomSparse(int blockRows, int blockCols, int BR, fProxy density, uint seed, Allocator allocator = Allocator.Temp)
        {
            if (blockRows < 1 || blockCols < 1) throw new ArgumentException("fProxyRandomSparse: block dims must be >= 1");
            if (blockRows < blockCols) throw new ArgumentException("fProxyRandomSparse: only square or tall (blockRows >= blockCols) is supported");
            if (BR < 1) throw new ArgumentException("fProxyRandomSparse: BR must be >= 1");

            var rng = new Unity.Mathematics.Random(seed == 0 ? 1u : seed);
            int diagCount = math.min(blockRows, blockCols);
            int deg = (int)math.round((double)density * blockCols) - 1;
            if (deg < 0) deg = 0;
            if (deg > blockCols - 1) deg = blockCols - 1;

            var offBound = new fProxyN(blockRows, Allocator.Temp);
            for (int i = 0; i < blockRows; i++) offBound[i] = (fProxy)0;

            int nnzbEstimate = diagCount + blockRows * (deg + 1);
            var builder = new fProxyBSRBuilder(blockRows, blockCols, BR, BR, Allocator.Temp, nnzbEstimate);

            var blk = new fProxyMxN(BR, BR, Allocator.Temp);
            fProxy blockRowBound = (fProxy)fProxySparseOffScale * BR;

            for (int i = 0; i < blockRows; i++)
            {
                for (int d = 0; d < deg; d++)
                {
                    int j = rng.NextInt(0, blockCols);
                    if (j == i && i < diagCount) continue;
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)/*+choose[rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale)|rng.NextDouble(-fProxySparseOffScale, fProxySparseOffScale)]*/rng.NextFloat(-fProxySparseOffScale, fProxySparseOffScale)/*-choose*/;
                    builder.AddBlock(i, j, in blk);
                    offBound[i] += blockRowBound;
                }
            }

            // See fProxyRandomSparseSPD for why the ref-dest Blas.dot primitive (manually
            // preallocated Di) is used here instead of the allocating convenience overload.
            var Mi = new fProxyMxN(BR, BR, Allocator.Temp);
            var Di = new fProxyMxN(BR, BR, Allocator.Temp, true);
            for (int i = 0; i < diagCount; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = (fProxy)rng.NextFloat(-1f, 1f);

                Blas.dot(in Mi, in Mi, ref Di, true);
                fProxy rho = (fProxy)(BR * BR) + offBound[i] + (fProxy)1;
                for (int dd = 0; dd < BR; dd++) Di[dd, dd] += rho;
                builder.AddBlock(i, i, in Di);
            }

            var result = builder.ToBSR(allocator);

            Di.Dispose();
            Mi.Dispose();
            blk.Dispose();
            builder.Dispose();
            offBound.Dispose();

            return result;
        }

        /// <summary>
        /// 2D grid (5-point) Dirichlet Laplacian on a <paramref name="gridX"/>×<paramref name="gridY"/>
        /// grid, as a block-tridiagonal <see cref="fProxyBSR"/> (block size BR = <paramref name="gridX"/>,
        /// one block-row per grid row, so N = gridX·gridY). SPD, and — unlike the diagonally-dominant
        /// random SPD generator — its spectrum is SPREAD (and, on a SQUARE grid, has exact symmetry
        /// MULTIPLICITIES), so it is the honest testbed for smallest-eigenpair solvers (LOBPCG) where the
        /// random generator's clustered bottom is misleading. The eigenvalues are analytic:
        /// <code>
        ///   λ(p,q) = (2 − 2·cos(pπ/(gridX+1))) + (2 − 2·cos(qπ/(gridY+1))),   p=1..gridX, q=1..gridY
        /// </code>
        /// The matrix is I⊗Tₓ + T_y⊗I with T = tridiag(−1, 2, −1): every diagonal block is
        /// tridiag(−1, 4, −1) (x-coupling + degree) and every between-block-row coupling is −I
        /// (y-coupling). On a square grid (gridX == gridY) λ(p,q) == λ(q,p) gives exact multiplicity-2
        /// pairs — the near-degenerate case guard vectors exist to handle. O(nnz), no dense form.
        /// </summary>
        /// <param name="gridX">Grid extent in x = the BSR block size BR (2–6 hit the unrolled spMV kernels).</param>
        /// <param name="gridY">Grid extent in y = the number of block rows.</param>
        public static fProxyBSR fProxyLaplacian2D(int gridX, int gridY, Allocator allocator = Allocator.Temp)
        {
            if (gridX < 1 || gridY < 1) throw new ArgumentException("fProxyLaplacian2D: grid dims must be >= 1");

            int BR = gridX, nb = gridY;
            int nnzbEstimate = nb + 2 * (nb - 1);
            if (nnzbEstimate < 1) nnzbEstimate = 1;
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp, nnzbEstimate);

            var D = new fProxyMxN(BR, BR, Allocator.Temp);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                    D[r, c] = (fProxy)0;
            for (int r = 0; r < BR; r++)
            {
                D[r, r] = (fProxy)4;
                if (r > 0)      D[r, r - 1] = (fProxy)(-1);
                if (r < BR - 1) D[r, r + 1] = (fProxy)(-1);
            }

            var C = new fProxyMxN(BR, BR, Allocator.Temp);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                    C[r, c] = (fProxy)(r == c ? -1 : 0);

            for (int i = 0; i < nb; i++)
            {
                builder.AddBlock(i, i, in D);
                if (i > 0)      builder.AddBlock(i, i - 1, in C);
                if (i < nb - 1) builder.AddBlock(i, i + 1, in C);
            }

            var result = builder.ToBSR(allocator);

            C.Dispose();
            D.Dispose();
            builder.Dispose();

            return result;
        }

        /// <summary>
        /// Penalty-pinned 3D grid truss stiffness matrix: (nx+1)×(ny+1)×(nz+1) nodes on a
        /// unit-spaced grid (nx × ny bays in plan, nz stories), 3 translational dof per node
        /// (N = 3·nodeCount; nx=ny=nz=1 gives the minimal N=24 unit cube). Assembled from axial
        /// bars (per-bar 3×3 stiffness blocks (EA/L)·uuᵀ): vertical columns on every grid line,
        /// X/Y beams plus one floor diagonal per panel at every level ≥ 1, and one wall brace per
        /// perimeter panel per story. Every base (level-0) node is pinned by adding
        /// <paramref name="penalty"/> to its three diagonal entries. Symmetric positive-definite
        /// for EA &gt; 0 and penalty &gt; 0; its conditioning grows with penalty/EA, making it the
        /// penalty-conditioned SPD stress case for single-precision eigensolvers. Returned as
        /// symmetric 3×3-block BSR; use <see cref="fProxyBSR.ToDense"/> for the dense form.
        /// </summary>
        /// <param name="nx">Bays in x (plan). Must be ≥ 1.</param>
        /// <param name="ny">Bays in y (plan). Must be ≥ 1.</param>
        /// <param name="nz">Stories (vertical bays). Must be ≥ 1.</param>
        /// <param name="EA">Axial bar stiffness (bar constant is EA/L). Must be &gt; 0.</param>
        /// <param name="penalty">Diagonal penalty pinning the base nodes. Must be &gt; 0.</param>
        public static fProxyBSR fProxyPenalizedGrid3D(int nx, int ny, int nz, fProxy EA, fProxy penalty, Allocator allocator = Allocator.Temp)
        {
            if (nx < 1 || ny < 1 || nz < 1)
                throw new ArgumentException("fProxyPenalizedGrid3D: nx/ny/nz must be >= 1");
            if (!(EA > (fProxy)0))
                throw new ArgumentException("fProxyPenalizedGrid3D: EA must be > 0");
            if (!(penalty > (fProxy)0))
                throw new ArgumentException("fProxyPenalizedGrid3D: penalty must be > 0");

            int NWx = nx + 1, NWy = ny + 1, levels = nz + 1;
            int nb = NWx * NWy * levels;

            int barCount = NWx * NWy * nz                      // columns
                         + nz * (nx * NWy + ny * NWx + nx * ny) // beams + floor diagonals
                         + nz * (2 * nx + 2 * ny);              // wall braces
            var builder = new fProxyBSRBuilder(nb, nb, 3, 3, Allocator.Temp, nb + 3 * barCount);

            int Node(int i, int j, int l) => (l * NWy + j) * NWx + i;

            void AddBar(int a, int b)
            {
                int ai = a % NWx, aj = (a / NWx) % NWy, al = a / (NWx * NWy);
                int bi = b % NWx, bj = (b / NWx) % NWy, bl = b / (NWx * NWy);
                fProxy dx = (fProxy)(bi - ai), dy = (fProxy)(bj - aj), dz = (fProxy)(bl - al);
                fProxy L = math.sqrt(dx * dx + dy * dy + dz * dz);
                fProxy ux = dx / L, uy = dy / L, uz = dz / L;
                fProxy kBar = EA / L;
                int lo = math.min(a, b), hi = math.max(a, b);
                for (int r = 0; r < 3; r++)
                {
                    fProxy ur = r == 0 ? ux : (r == 1 ? uy : uz);
                    for (int c = 0; c < 3; c++)
                    {
                        fProxy uc = c == 0 ? ux : (c == 1 ? uy : uz);
                        fProxy v = kBar * ur * uc;
                        builder.AddValue(3 * a + r, 3 * a + c, v);
                        builder.AddValue(3 * b + r, 3 * b + c, v);
                        builder.AddValue(3 * hi + r, 3 * lo + c, -v);
                    }
                }
            }

            // vertical columns on every grid line
            for (int l = 0; l < nz; l++)
                for (int j = 0; j < NWy; j++)
                    for (int i = 0; i < NWx; i++)
                        AddBar(Node(i, j, l), Node(i, j, l + 1));

            // X/Y beams + one floor diagonal per panel, at every level >= 1
            for (int l = 1; l <= nz; l++)
            {
                for (int j = 0; j < NWy; j++)
                    for (int i = 0; i < nx; i++)
                        AddBar(Node(i, j, l), Node(i + 1, j, l));
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < NWx; i++)
                        AddBar(Node(i, j, l), Node(i, j + 1, l));
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++)
                        AddBar(Node(i, j, l), Node(i + 1, j + 1, l));
            }

            // one brace per perimeter wall panel per story (all rising one story across one bay)
            for (int s = 0; s < nz; s++)
            {
                for (int i = 0; i < nx; i++)
                {
                    AddBar(Node(i, 0, s), Node(i + 1, 0, s + 1));
                    AddBar(Node(i, ny, s), Node(i + 1, ny, s + 1));
                }
                for (int j = 0; j < ny; j++)
                {
                    AddBar(Node(0, j, s), Node(0, j + 1, s + 1));
                    AddBar(Node(nx, j, s), Node(nx, j + 1, s + 1));
                }
            }

            // penalty pins: every base (level-0) node's three diagonal entries
            for (int j = 0; j < NWy; j++)
                for (int i = 0; i < NWx; i++)
                {
                    int node = Node(i, j, 0);
                    for (int d = 0; d < 3; d++)
                        builder.AddValue(3 * node + d, 3 * node + d, penalty);
                }

            var result = builder.ToBSRSymmetric(allocator);
            builder.Dispose();
            return result;
        }
    }
}
