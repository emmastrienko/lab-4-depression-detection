using Lab4.Shared.Models;
using Lab4.Shared.Onnx;
using Lab4.Shared.Text;
using Lab4.Web.Models;
using Microsoft.Extensions.ML;
using Microsoft.ML;

namespace Lab4.Web.Endpoints;

// Реєстрація HTTP-ендпоінтів передбачення та керування моделлю
public static class PredictionEndpoints
{
    public const string ModelName = "DepressionModel";

    public static void MapPredictionEndpoints(this WebApplication app, string onnxPath, string zipPath)
    {
        // Передбачення для одного тексту
        app.MapPost("/api/predict", (PredictRequest req, PredictionEnginePool<TweetInput, OnnxPrediction> pool) =>
        {
            if (string.IsNullOrWhiteSpace(req?.Text))
                return Results.BadRequest(new { error = "Поле text порожнє" });

            var p = pool.Predict(ModelName, new TweetInput { Text = TextCleaner.Clean(req.Text) });
            return Results.Ok(new PredictResponse(req.Text, p.PredictedLabel, p.Prob, p.ScoreValue));
        });

        // Передбачення для кількох текстів
        app.MapPost("/api/predict/batch", (string[] texts, PredictionEnginePool<TweetInput, OnnxPrediction> pool) =>
            Results.Ok(texts.Select(t =>
            {
                var p = pool.Predict(ModelName, new TweetInput { Text = TextCleaner.Clean(t) });
                return new PredictResponse(t, p.PredictedLabel, p.Prob, p.ScoreValue);
            })));

        // Повторний імпорт ONNX (наприклад, після перенавчання в Lab4.Train).
        // Пул підхопить новий zip автоматично завдяки watchForChanges
        app.MapPost("/api/model/reload", () =>
        {
            OnnxModelImporter.ImportAndSave(new MLContext(), onnxPath, zipPath);
            return Results.Ok(new { reloaded = true, onnx = onnxPath, updated = File.GetLastWriteTime(onnxPath) });
        });

        app.MapGet("/api/model", () => Results.Ok(new
        {
            name = ModelName,
            onnx = onnxPath,
            sizeKb = Math.Round(new FileInfo(onnxPath).Length / 1024.0, 1),
            updated = File.GetLastWriteTime(onnxPath)
        }));
    }
}
