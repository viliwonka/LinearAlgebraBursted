using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Procedural terrain for <see cref="HoverTankMPCDemo"/>: a layered-noise height field plus the
    /// mesh, renderer and <see cref="MeshCollider"/> built from it. Sceneless — no prefab, no asset,
    /// everything is made in code so dropping the demo component on an empty GameObject is enough.
    ///
    /// <see cref="Height"/> is TERRAIN TRUTH. It is here to build the mesh and to measure how far an
    /// estimate has drifted from reality. SENSORS MUST NOT CALL IT: a sensor raycasts against the
    /// collider the way a real range finder does, and what it can and cannot see — occlusion, misses
    /// past a drop-off, nothing at all beyond <see cref="Size"/> — is part of the problem being posed.
    /// Height is deterministic and free of managed state, so it is callable from a Burst job.
    ///
    /// The field is shaped rather than left as plain noise: a FLAT APRON out to
    /// <see cref="ApronRadius"/> around the origin so the tank spawns on level ground, ROLLING HILLS
    /// whose summed gradient stays gentle enough for the hover loop to track, one STEEP FACE (the
    /// escarpment across the north) and one near-vertical WALL.
    /// </summary>
    public static class TerrainField
    {
        /// <summary>Extent on each horizontal axis, metres: the field spans [-Size/2, +Size/2].</summary>
        public const float Size = 200f;

        /// <summary>Mesh cells per axis. Cell size is <see cref="Size"/> / <see cref="Cells"/>.</summary>
        public const int Cells = 200;

        /// <summary>Radius around the origin that is exactly flat, metres.</summary>
        public const float ApronRadius = 16f;

        /// <summary>Radius at which the hills reach full amplitude, metres.</summary>
        public const float ApronBlend = 42f;

        // Rolling hills, three octaves. AMPLITUDE PER WAVELENGTH is the number that matters, not
        // amplitude: it sets the gradient, and the gradient is what the hover loop mistakes for hull
        // tilt, so these are picked to keep the hills near 10 deg typical and under 30 deg anywhere,
        // taking their relief from a long base wavelength rather than from a steeper one. The apron
        // blend is wide for the same reason — it is itself a slope, of roughly
        // 1.5 * amplitude / (ApronBlend - ApronRadius).
        const float HillAmp0 = 4.00f, HillLen0 = 100f;
        const float HillAmp1 = 1.05f, HillLen1 = 30f;
        const float HillAmp2 = 0.32f, HillLen2 = 11f;

        /// <summary>South and north edge of the escarpment's face, metres.</summary>
        public const float EscarpStartZ = 44f, EscarpEndZ = 60f;

        /// <summary>Height gained from the foot of the escarpment to the plateau, metres.</summary>
        public const float EscarpRise = 7f;

        /// <summary>Centre line of the wall, metres.</summary>
        public const float WallCenterX = 0f, WallCenterZ = -34f;

        /// <summary>Wall half-extents measured to where it is still at full height, metres.</summary>
        public const float WallHalfLength = 16f, WallHalfThick = 2.5f;

        /// <summary>Horizontal run the wall takes to reach full height, metres.</summary>
        public const float WallFade = 1.5f;

        /// <summary>Wall height above the local ground, metres.</summary>
        public const float WallHeight = 7f;

        /// <summary>
        /// Terrain height at world (x, z), metres. Zero over the apron. Defined everywhere, including
        /// outside the meshed field — the mesh, and therefore anything that raycasts, stops at
        /// <see cref="Size"/>.
        /// </summary>
        public static float Height(float x, float z)
        {
            float2 p = new float2(x, z);

            float hills = HillAmp0 * noise.snoise(p * (1f / HillLen0) + new float2(17.3f, -41.9f))
                        + HillAmp1 * noise.snoise(p * (1f / HillLen1) + new float2(-53.7f, 88.1f))
                        + HillAmp2 * noise.snoise(p * (1f / HillLen2) + new float2(91.1f, 12.7f));

            // math.smoothstep is exactly 0 below its first edge, so the apron is exactly flat rather
            // than nearly flat, and the tank's first frames are not fighting a gradient.
            float h = hills * math.smoothstep(ApronRadius, ApronBlend, math.length(p));

            h += EscarpRise * math.smoothstep(EscarpStartZ, EscarpEndZ, z);

            float alongX = 1f - math.smoothstep(WallHalfLength - WallFade, WallHalfLength + WallFade,
                                                math.abs(x - WallCenterX));
            float acrossZ = 1f - math.smoothstep(WallHalfThick - WallFade, WallHalfThick + WallFade,
                                                 math.abs(z - WallCenterZ));
            h += WallHeight * alongX * acrossZ;

            return h;
        }

        /// <summary>Terrain height at a world XZ point, metres.</summary>
        public static float Height(float2 p) => Height(p.x, p.y);

        /// <summary>
        /// Builds the terrain GameObject centred on the origin: the height-field mesh, a renderer and
        /// the <see cref="MeshCollider"/> the sense rays need. Caller owns the returned object.
        ///
        /// Starts from a primitive purely so the renderer inherits the ACTIVE RENDER PIPELINE's default
        /// material; both its render mesh and its collider mesh are then replaced.
        /// </summary>
        public static GameObject Build(string name)
        {
            const int verts = Cells + 1;
            const float step = Size / Cells;
            const float half = Size * 0.5f;

            var vertices = new Vector3[verts * verts];
            var uv = new Vector2[verts * verts];
            var tris = new int[Cells * Cells * 6];

            for (int j = 0; j < verts; j++)
            {
                float z = -half + j * step;
                for (int i = 0; i < verts; i++)
                {
                    float x = -half + i * step;
                    int v = j * verts + i;
                    vertices[v] = new Vector3(x, Height(x, z), z);
                    uv[v] = new Vector2(i * step * 0.1f, j * step * 0.1f);
                }
            }

            int t = 0;
            for (int j = 0; j < Cells; j++)
                for (int i = 0; i < Cells; i++)
                {
                    int v = j * verts + i;
                    tris[t++] = v; tris[t++] = v + verts; tris[t++] = v + 1;
                    tris[t++] = v + 1; tris[t++] = v + verts; tris[t++] = v + verts + 1;
                }

            var mesh = new Mesh { name = "HoverTankMPC_TerrainMesh" };
            // Set before the buffers: raising Cells past a 256x256 grid overflows a 16-bit index.
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.vertices = vertices;
            mesh.uv = uv;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var go = GameObject.CreatePrimitive(PrimitiveType.Plane);
            go.name = name;
            go.transform.localScale = Vector3.one;
            go.GetComponent<MeshFilter>().sharedMesh = mesh;
            go.GetComponent<MeshCollider>().sharedMesh = mesh;
            go.GetComponent<Renderer>().material.color = new Color(0.42f, 0.45f, 0.36f);
            return go;
        }
    }
}
