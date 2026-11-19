using Microsoft.ML.Data;

namespace Lab4.Shared.Models;

// Вихід моделі ML.NET (zip)
public class TweetPrediction
{
    [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
