using Lab4.Shared.Models;
using Microsoft.ML;

namespace Lab4.Train.Training;

// Конвеєр із трансформів, що підтримують експорт в ONNX
// (FeaturizeText експорт в ONNX не підтримує, тому розкладено на кроки)
public static class ExportablePipelineBuilder
{
    private static readonly string[] StopWords =
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "from",
        "in", "on", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
        "those", "am", "as", "do", "does", "did", "so", "than", "too", "very", "just", "there", "here", "then",
        "s", "t", "ll", "m", "re", "ve", "d"
    };

    public static IEstimator<ITransformer> Build(MLContext ml)
    {
        return ml.Transforms.Text.TokenizeIntoWords("Tokens", nameof(TweetData.Text))
            .Append(ml.Transforms.Text.RemoveStopWords("Tokens", "Tokens", StopWords))
            .Append(ml.Transforms.Conversion.MapValueToKey("TokenKeys", "Tokens"))
            .Append(ml.Transforms.Text.ProduceNgrams("Ngrams", "TokenKeys", ngramLength: 2, useAllLengths: true))
            .Append(ml.Transforms.NormalizeLpNorm("Features", "Ngrams"))
            .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(TweetData.Label), featureColumnName: "Features"));
    }
}
