// Лабораторна робота 4. Робота з ONNX та побудова додатків
// На основі ЛР 2 (варіант 5): бінарна класифікація твітів — є ознаки депресії чи ні.
// Консольний додаток: навчання -> оцінка -> збереження zip -> експорт ONNX -> імпорт ONNX і перевірка.

using Lab4.Shared.Models;
using Lab4.Shared.Onnx;
using Lab4.Train.Data;
using Lab4.Train.Evaluation;
using Lab4.Train.Onnx;
using Lab4.Train.Reporting;
using Lab4.Train.Training;
using Microsoft.ML;

var dataPath = args.Length > 0 ? args[0] : Path.Combine("data", "sentiment_tweets3.csv");
if (!File.Exists(dataPath))
{
    Console.WriteLine($"Файл не знайдено: {Path.GetFullPath(dataPath)}");
    Console.WriteLine("Скопіюйте sentiment_tweets3.csv з ЛР 2 у папку data/ або передайте шлях аргументом.");
    return;
}

var modelsDir = Path.GetFullPath(Path.Combine("..", "Models"));
Directory.CreateDirectory(modelsDir);
var zipPath = Path.Combine(modelsDir, "depression_model.zip");
var onnxPath = Path.Combine(modelsDir, "depression_model.onnx");

var ml = new MLContext(seed: 42);

// 1. Завантаження та очищення даних
var cleaned = new TweetDataLoader(ml).Load(dataPath);
ConsoleReporter.PrintClassBalance(cleaned);

var data = ml.Data.LoadFromEnumerable(cleaned);
var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 42);

// 2. Конвеєр із трансформів, що підтримують експорт в ONNX
var pipeline = ExportablePipelineBuilder.Build(ml);

// 3. Навчання та оцінка
var sw = System.Diagnostics.Stopwatch.StartNew();
var model = pipeline.Fit(split.TrainSet);
sw.Stop();
ConsoleReporter.PrintTrained(sw.Elapsed.TotalSeconds);
ConsoleReporter.PrintMetrics(ml.BinaryClassification.Evaluate(model.Transform(split.TestSet)));

// 4. Збереження у форматі ML.NET (zip)
ml.Model.Save(model, split.TrainSet.Schema, zipPath);
Console.WriteLine($"Модель ML.NET збережено: {zipPath}");

var exporter = new OnnxExporter(ml);

// 5. Експорт в ONNX
exporter.Export(model, onnxPath);
Console.WriteLine($"Модель експортовано в ONNX: {onnxPath} ({new FileInfo(onnxPath).Length / 1024.0:F1} KB)");

// 6. Входи/виходи ONNX-графа
exporter.PrintGraphSchema(onnxPath);

// 7. Імпорт ONNX через ApplyOnnxModel
var onnxModel = OnnxModelImporter.Import(ml, onnxPath);
ConsoleReporter.PrintOutputSchema(onnxModel);

// 8. Порівняння ML.NET і ONNX на тестовій вибірці
var testTexts = ml.Data.CreateEnumerable<TweetData>(split.TestSet, reuseRowObject: false).ToList();
var comparison = new OnnxComparator(ml).Compare(model, onnxModel, testTexts);
ConsoleReporter.PrintComparison(comparison);

// 9. Передбачення для власних прикладів (ML.NET vs ONNX)
string[] samples =
{
    "I feel so empty and hopeless, nothing matters anymore",
    "Had an amazing day at the beach with my friends!",
    "I can't sleep again, I'm tired of everything and everyone",
    "Just finished my project, really proud of myself today",
    "My depression is getting worse and I don't know who to talk to"
};
var predictor = new OnnxSamplePredictor(ml, model, onnxModel);
ConsoleReporter.PrintSamplePredictions(predictor, samples);
