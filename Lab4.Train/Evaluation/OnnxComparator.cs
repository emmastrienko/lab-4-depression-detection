using Lab4.Shared.Models;
using Microsoft.ML;

namespace Lab4.Train.Evaluation;

// Порівняння результатів ML.NET та ONNX на тестовій вибірці
public sealed record OnnxComparison(int Total, int SameClass, int OnnxCorrect, double MaxProbabilityDiff, double Seconds);

public class OnnxComparator
{
    private readonly MLContext _ml;

    public OnnxComparator(MLContext ml)
    {
        _ml = ml;
    }

    public OnnxComparison Compare(ITransformer mlnetModel, ITransformer onnxModel, List<TweetData> testSet)
    {
        var netEngine = _ml.Model.CreatePredictionEngine<TweetInput, TweetPrediction>(mlnetModel);
        var onnxEngine = _ml.Model.CreatePredictionEngine<TweetInput, OnnxPrediction>(onnxModel);

        int same = 0, onnxCorrect = 0;
        double maxDiff = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (var t in testSet)
        {
            var input = new TweetInput { Text = t.Text };
            var a = netEngine.Predict(input);
            var b = onnxEngine.Predict(input);
            if (a.PredictedLabel == b.PredictedLabel) same++;
            if (b.PredictedLabel == t.Label) onnxCorrect++;
            maxDiff = Math.Max(maxDiff, Math.Abs(a.Probability - b.Prob));
        }
        sw.Stop();

        return new OnnxComparison(testSet.Count, same, onnxCorrect, maxDiff, sw.Elapsed.TotalSeconds);
    }
}
