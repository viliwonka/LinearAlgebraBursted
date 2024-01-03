using Unity.Burst;

namespace LinearAlgebra.Stats
{
    public struct doubleMeanMinMaxRangeStats
    {
        public double mean;
        public double min;
        public double max;
        public double range;

        public doubleMeanMinMaxRangeStats(double mean, double min, double max, double range)
        {
            this.mean = mean;
            this.min = min;
            this.max = max;
            this.range = range;
        }
    }

    [BurstCompile]
    public struct doubleFullStats
    {
        public double count;
        public double mean;
        public double min;
        public double max;
        public double range;
        public double median; // Q2
        public double stdDev;
        public double variance;
        public double iqr;
        public double q1;
        public double q3;

        public doubleFullStats(double count, double mean, double min, double max, double range, double median, double stdDev, double variance, double iqr, double q1, double q3)
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