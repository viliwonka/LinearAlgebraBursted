using Unity.Burst;

namespace LinearAlgebra.Stats
{
    public struct fProxyMeanMinMaxRangeStats
    {
        public fProxy mean;
        public fProxy min;
        public fProxy max;
        public fProxy range;

        public fProxyMeanMinMaxRangeStats(fProxy mean, fProxy min, fProxy max, fProxy range)
        {
            this.mean = mean;
            this.min = min;
            this.max = max;
            this.range = range;
        }
    }

    [BurstCompile]
    public struct fProxyFullStats
    {
        public fProxy count;
        public fProxy mean;
        public fProxy min;
        public fProxy max;
        public fProxy range;
        public fProxy median; // Q2
        public fProxy stdDev;
        public fProxy variance;
        public fProxy iqr;
        public fProxy q1;
        public fProxy q3;

        public fProxyFullStats(fProxy count, fProxy mean, fProxy min, fProxy max, fProxy range, fProxy median, fProxy stdDev, fProxy variance, fProxy iqr, fProxy q1, fProxy q3)
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