using Microsoft.ML.Data;

namespace Lab4.Shared.Models;

// Вихід ONNX-моделі: ML.NET додає до назв вихідних колонок суфікс ".output",
// а ONNX повертає тензори, тому типи — масиви
public class OnnxPrediction
{
    // ML.NET знімає суфікс ".output" з назв виходів ONNX-моделі, експортованої з ML.NET
    [ColumnName("Probability")] public float[] Probability { get; set; }
    [ColumnName("Score")] public float[] Score { get; set; }

    [NoColumn] public float Prob => Probability?.Length > 0 ? Probability[0] : float.NaN;
    [NoColumn] public float ScoreValue => Score?.Length > 0 ? Score[0] : float.NaN;
    [NoColumn] public bool PredictedLabel => Prob >= 0.5f;
}
