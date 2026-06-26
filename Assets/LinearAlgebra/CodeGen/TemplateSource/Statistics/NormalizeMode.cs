//singularFile//
namespace LinearAlgebra
{
    // Feature-scaling mode for StatsOP.normalizeColumns / normalizeRows.
    // MinMax: maps each axis to [0,1] via (x − min)/(max − min).
    //         Constant axis (max == min) → all entries set to 0.
    // ZScore: standardises each axis via (x − mean)/stdDev using the POPULATION
    //         std dev (÷N, not ÷(N−1)).  Constant axis (stdDev == 0) → all entries set to 0.
    public enum NormalizeMode { MinMax, ZScore }
}
