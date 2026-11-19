using Lab4.Shared.Models;
using Lab4.Train.Evaluation;
using Lab4.Train.Onnx;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Lab4.Train.Reporting;

// Виведення результатів у консоль
public static class ConsoleReporter
{
    public static void PrintClassBalance(List<TweetData> data)
    {
        int pos = data.Count(t => t.Label);
        Console.WriteLine($"Записів: {data.Count}; депресія (1): {pos} ({100.0 * pos / data.Count:F1}%)");
    }

    public static void PrintTrained(double seconds)
    {
        Console.WriteLine($"\nМодель навчено (SdcaLogisticRegression) за {seconds:F1} с");
    }

    public static void PrintMetrics(CalibratedBinaryClassificationMetrics m)
    {
        Console.WriteLine($"Accuracy={m.Accuracy:P2}  AUC={m.AreaUnderRocCurve:F3}  Precision={m.PositivePrecision:P2}  " +
                          $"Recall={m.PositiveRecall:P2}  F1={m.F1Score:F3}  LogLoss={m.LogLoss:F3}");
        Console.WriteLine(m.ConfusionMatrix.GetFormattedConfusionTable());
    }

    public static void PrintOutputSchema(ITransformer onnxModel)
    {
        var ml = new MLContext();
        var emptyData = ml.Data.LoadFromEnumerable(new List<TweetInput>());

        Console.WriteLine("\nСхема виходу ONNX-трансформера в ML.NET:");
        foreach (var col in onnxModel.GetOutputSchema(emptyData.Schema))
            Console.WriteLine($"  {col.Name}: {col.Type}");
    }

    public static void PrintComparison(OnnxComparison c)
    {
        Console.WriteLine($"\nПеревірка ONNX на {c.Total} тестових записах ({c.Seconds:F1} с):");
        Console.WriteLine($"  Збіг класів ML.NET та ONNX: {c.SameClass}/{c.Total} ({100.0 * c.SameClass / c.Total:F2}%)");
        Console.WriteLine($"  Accuracy ONNX: {100.0 * c.OnnxCorrect / c.Total:F2}%");
        Console.WriteLine($"  Макс. різниця ймовірностей: {c.MaxProbabilityDiff:E2}");
    }

    public static void PrintSamplePredictions(OnnxSamplePredictor predictor, string[] samples)
    {
        Console.WriteLine("\nПрогнози для власних прикладів:");
        Console.WriteLine($"  {"ML.NET",-10}{"P(ML.NET)",10}{"ONNX",10}{"P(ONNX)",10}  Текст");
        foreach (var (text, a, b) in predictor.PredictSamples(samples))
            Console.WriteLine($"  {(a.PredictedLabel ? "депресія" : "норма"),-10}{a.Probability,10:F3}" +
                              $"{(b.PredictedLabel ? "депресія" : "норма"),10}{b.Prob,10:F3}  {text}");
    }
}
