//singularFile//
namespace LinearAlgebra.ML
{
    // Seeding strategy for fProxyKMeansOP.kmeans.
    // KMeansPlusPlus = D²-weighted seeding (Arthur & Vassilvitskii 2007); O(k²·N·D);
    //   improves convergence and solution quality significantly over random init.
    // Uniform = pick k distinct random points via reservoir selection; O(N); fast
    //   for large k or when budget is tight and init quality matters less.
    public enum KMeansInit { KMeansPlusPlus, Uniform }
}
