// Лабораторна робота 4. Веб-додаток (ASP.NET Core Minimal API)
// Імпорт моделі з ONNX + реєстрація через AddPredictionEnginePool.

using Lab4.Shared.Models;
using Lab4.Shared.Onnx;
using Lab4.Web.Endpoints;
using Microsoft.Extensions.ML;
using Microsoft.ML;

var builder = WebApplication.CreateBuilder(args);

var root = builder.Environment.ContentRootPath;
var onnxPath = Path.GetFullPath(Path.Combine(root, builder.Configuration["Model:OnnxPath"]));
var zipPath = Path.GetFullPath(Path.Combine(root, builder.Configuration["Model:ZipPath"]));

if (!File.Exists(onnxPath))
    throw new FileNotFoundException("ONNX-модель не знайдено. Спочатку запустіть Lab4.Train.", onnxPath);

// 1. Імпорт ONNX: ApplyOnnxModel + Fit на порожніх даних,
//    результат зберігається у zip, щоб його міг завантажити PredictionEnginePool
OnnxModelImporter.ImportAndSave(new MLContext(), onnxPath, zipPath);

// 2. Реєстрація моделі: пул PredictionEngine (потокобезпечний, на відміну від одного PredictionEngine).
//    watchForChanges: при оновленні zip модель перезавантажується без перезапуску додатку
builder.Services.AddPredictionEnginePool<TweetInput, OnnxPrediction>()
    .FromFile(modelName: PredictionEndpoints.ModelName, filePath: zipPath, watchForChanges: true);

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPredictionEndpoints(onnxPath, zipPath);

app.Run();
