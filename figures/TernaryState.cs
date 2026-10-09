// TernaryState.cs
// A point that moves inside a ternary plot (three shares A, B, C that always sum to 1),
// plus everything you can derive from it: camp, intensity, lean and a sortable key.
//
// Plain C# with no Unity dependency, so it drops into a Unity project (any version with
// .NET Standard 2.0+) or any other .NET program. Mirrors ternary_index/state.py and
// ternary_index/geometry.py in this repository; see the README for the math.
//
// Stored state:  A, B, C (the shares) and Actions (how many actions have been applied).
// Everything else is computed on demand, so there is nothing to keep in sync.

using System;

namespace TernaryIndexing
{
    /// <summary>Which region of the triangle a point is in.</summary>
    public enum Camp
    {
        Neutral = 0,   // close to the center (intensity below NeutralRadius)
        A = 1,         // A is the largest share
        B = 2,
        C = 3
    }

    /// <summary>How "far from the center" is measured.</summary>
    public enum IntensityMetric
    {
        /// <summary>Straight-line distance from the center; 1 only at the three apexes,
        /// 0.5 at edge midpoints.</summary>
        Euclidean,
        /// <summary>1 - 3 * min share; 1 anywhere on the outer edge.</summary>
        Edge
    }

    [Serializable]
    public class TernaryState
    {
        // ------------------------------------------------------------------ stored state
        public float A = 1f / 3f;
        public float B = 1f / 3f;
        public float C = 1f / 3f;
        public int Actions = 0;

        // ------------------------------------------------------------------ tuning
        /// <summary>Overall responsiveness. 1 = each action counts as one tally.</summary>
        public float Step = 1f;
        /// <summary>Imaginary balanced actions at the start. Larger = steadier early game.</summary>
        public float Seed = 4f;
        /// <summary>Intensity below which the point counts as Neutral.</summary>
        public float NeutralRadius = 0.15f;
        public IntensityMetric Metric = IntensityMetric.Euclidean;
        /// <summary>Width of each camp's key range (Key = camp * Band + level).</summary>
        public int Band = 100;

        /// <summary>Raised after an action moves the point into a different camp (old, new).</summary>
        public event Action<Camp, Camp> CampChanged;

        public TernaryState() { }

        public TernaryState(float a, float b, float c, int actions = 0)
        {
            Set(a, b, c);
            Actions = actions;
        }

        // ------------------------------------------------------------------ updates
        /// <summary>Fraction of the way the next action moves toward its target.
        /// Shrinks as Actions grows: early actions matter more than late ones.</summary>
        public float Alpha
        {
            get { return Math.Min(1f, Step / (Seed + Actions + 1f)); }
        }

        /// <summary>Pull the point toward a target mix (ta, tb, tc). The target is
        /// normalized, so (2, 1, 0) means "two parts A, one part B".
        /// weight scales just this action (e.g. 2 for a major choice).</summary>
        public void Apply(float ta, float tb, float tc, float weight = 1f)
        {
            float sum = ta + tb + tc;
            if (sum <= 0f) throw new ArgumentException("target shares must sum to > 0");
            ta /= sum; tb /= sum; tc /= sum;

            Camp before = CurrentCamp;
            float alpha = Math.Min(1f, Alpha * weight);
            Set(A + alpha * (ta - A), B + alpha * (tb - B), C + alpha * (tc - C));
            Actions++;
            Camp after = CurrentCamp;
            if (after != before && CampChanged != null) CampChanged(before, after);
        }

        /// <summary>A pure action toward one apex.</summary>
        public void Toward(Camp corner, float weight = 1f)
        {
            switch (corner)
            {
                case Camp.A: Apply(1, 0, 0, weight); break;
                case Camp.B: Apply(0, 1, 0, weight); break;
                case Camp.C: Apply(0, 0, 1, weight); break;
                default: Balance(weight); break;
            }
        }

        /// <summary>A mixed action halfway between two apexes.</summary>
        public void Blend(Camp first, Camp second, float weight = 1f)
        {
            float a = 0, b = 0, c = 0;
            Add(first, ref a, ref b, ref c);
            Add(second, ref a, ref b, ref c);
            Apply(a, b, c, weight);
        }

        /// <summary>A balancing action: pull toward the exact center.</summary>
        public void Balance(float weight = 1f)
        {
            Apply(1, 1, 1, weight);
        }

        public void Reset()
        {
            A = B = C = 1f / 3f;
            Actions = 0;
        }

        // ------------------------------------------------------------------ derived values
        /// <summary>Straight-line distance from the center; apex = 1, edge midpoint = 0.5.</summary>
        public float Magnitude { get { return MagnitudeOf(A, B, C); } }

        /// <summary>1 - 3 * min share; 1 anywhere on the outer edge.</summary>
        public float EdgeIntensity { get { return EdgeIntensityOf(A, B, C); } }

        /// <summary>Intensity under the chosen Metric.</summary>
        public float Intensity { get { return IntensityOf(A, B, C, Metric); } }

        public Camp CurrentCamp { get { return CampOf(A, B, C, NeutralRadius, Metric); } }

        /// <summary>0 = pointing straight at its own apex, 1 = on the border with the
        /// neighboring camp (two largest shares tied).</summary>
        public float Lean { get { return LeanOf(A, B, C); } }

        /// <summary>The camp this point leans toward (second-largest share).</summary>
        public Camp LeaningToward { get { return SecondOf(A, B, C); } }

        /// <summary>camp * Band + level. Same camp = same leading digits; within a camp the
        /// key rises with intensity. Example (Band 100): A camp at intensity 0.91 = 191.</summary>
        public int Key { get { return KeyOf(A, B, C, NeutralRadius, Metric, Band); } }

        public override string ToString()
        {
            return string.Format("A {0:0.000}  B {1:0.000}  C {2:0.000}  camp {3}  intensity {4:0.000}  " +
                                 "lean {5:0.00}->{6}  key {7}  actions {8}",
                                 A, B, C, CurrentCamp, Intensity, Lean, LeaningToward, Key, Actions);
        }

        // ------------------------------------------------------------------ static helpers
        // Usable on any triplet, e.g. to tag content with the position it belongs to.

        public static float MagnitudeOf(float a, float b, float c)
        {
            const float third = 1f / 3f;
            float da = a - third, db = b - third, dc = c - third;
            return (float)Math.Sqrt(1.5 * (da * da + db * db + dc * dc));
        }

        public static float EdgeIntensityOf(float a, float b, float c)
        {
            return 1f - 3f * Math.Min(a, Math.Min(b, c));
        }

        public static float IntensityOf(float a, float b, float c, IntensityMetric metric)
        {
            return metric == IntensityMetric.Euclidean ? MagnitudeOf(a, b, c) : EdgeIntensityOf(a, b, c);
        }

        /// <summary>Largest share wins; ties go to the earlier letter.</summary>
        public static Camp LeaderOf(float a, float b, float c)
        {
            if (a >= b && a >= c) return Camp.A;
            if (b >= c) return Camp.B;
            return Camp.C;
        }

        public static Camp CampOf(float a, float b, float c, float neutralRadius, IntensityMetric metric)
        {
            return IntensityOf(a, b, c, metric) < neutralRadius ? Camp.Neutral : LeaderOf(a, b, c);
        }

        public static float LeanOf(float a, float b, float c)
        {
            float hi = Math.Max(a, Math.Max(b, c));
            float lo = Math.Min(a, Math.Min(b, c));
            float mid = a + b + c - hi - lo;
            float span = hi - lo;
            return span > 1e-6f ? (mid - lo) / span : 0f;
        }

        public static Camp SecondOf(float a, float b, float c)
        {
            // Stable ascending sort of the three indices (ties keep A before B before C),
            // then take the middle one. Matches numpy's argsort(w)[1] in the Python version.
            float[] v = { a, b, c };
            int[] order = { 0, 1, 2 };
            for (int pass = 0; pass < 2; pass++)
                for (int i = 0; i < 2; i++)
                    if (v[order[i]] > v[order[i + 1]])
                    {
                        int t = order[i]; order[i] = order[i + 1]; order[i + 1] = t;
                    }
            return (Camp)(order[1] + 1);
        }

        public static int KeyOf(float a, float b, float c, float neutralRadius, IntensityMetric metric, int band)
        {
            // The tiny epsilon keeps exact band edges (e.g. intensity 0.25) from flooring down
            // because of float rounding; the Python version uses the same epsilon.
            int level = (int)Math.Floor(band * (double)IntensityOf(a, b, c, metric) + 1e-4);
            if (level > band - 1) level = band - 1;
            if (level < 0) level = 0;
            return (int)CampOf(a, b, c, neutralRadius, metric) * band + level;
        }

        // ------------------------------------------------------------------ internals
        private void Set(float a, float b, float c)
        {
            a = Math.Max(0f, a); b = Math.Max(0f, b); c = Math.Max(0f, c);
            float sum = a + b + c;                      // guard against float drift
            A = a / sum; B = b / sum; C = c / sum;
        }

        private static void Add(Camp corner, ref float a, ref float b, ref float c)
        {
            if (corner == Camp.A) a += 1;
            else if (corner == Camp.B) b += 1;
            else if (corner == Camp.C) c += 1;
            else { a += 1f / 3f; b += 1f / 3f; c += 1f / 3f; }
        }
    }
}
