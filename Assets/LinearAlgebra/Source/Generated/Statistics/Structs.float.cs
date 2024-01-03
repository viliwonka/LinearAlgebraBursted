using Unity.Burst;

namespace LinearAlgebra.Stats
{
    public struct floatMeanMinMaxRangeStats
    {
        public float mean;
        public float min;
        public float max;
        public float range;

        public floatMeanMinMaxRangeStats(float mean, float min, float max, float range)
        {
            this.mean = mean;
            this.min = min;
            this.max = max;
            this.range = range;
        }
    }

    [BurstCompile]
    public struct floatFullStats
    {
        public float count;
        public float mean;
        public float min;
        public float max;
        public float range;
        public float median; // Q2
        public float stdDev;
        public float variance;
        public float iqr;
        public float q1;
        public float q3;

        public floatFullStats(float count, float mean, float min, float max, float range, float median, float stdDev, float variance, float iqr, float q1, float q3)
        {
            this.count = count;
            this.mean = mean;
            this.min = min;
            this.max = max;
            this.range = range;
            this.median = median;
            this.stdDev = stdDev;
            this.variance = variance;
            this.iqr = iqr;
            this.q1 = q1;
            this.q3 = q3;
        }

        public override string ToString()
        {
            return $"Mean: {mean}, Min: {min}, Max: {max}, Range: {range}, Median: {median}, StdDev: {stdDev}, Variance: {variance}, IQR: {iqr}, Q1: {q1}, Q3: {q3}";
        }
    }
}