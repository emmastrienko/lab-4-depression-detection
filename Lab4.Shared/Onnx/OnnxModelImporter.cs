using Lab4.Shared.Models;
using Microsoft.ML;

namespace Lab4.Shared.Onnx;

// Імпорт ONNX-моделі в ML.NET та збереження трансформера у zip
public static class OnnxModelImporter
{
    // Імпорт ONNX через ApplyOnnxModel. Fit на порожніх даних лише створює трансформер
    public static ITransformer Import(MLContext ml, string onnxPath)
    {
        var emptyData = ml.Data.LoadFromEnumerable(new List<TweetInput>());
        return ml.Transforms
            .ApplyOnnxModel(OnnxNames.Outputs, OnnxNames.Inputs, onnxPath, OnnxNames.Shapes,
                gpuDeviceId: null, fallbackToCpu: false, recursionLimit: 100)
            .Fit(emptyData);
    }

    // Імпорт + збереження у zip для PredictionEnginePool.
    // Запис через тимчасовий файл, щоб пул не прочитав напівзаписаний zip
    public static void ImportAndSave(MLContext ml, string onnxPath, string zipPath)
    {
        var emptyData = ml.Data.LoadFromEnumerable(new List<TweetInput>());
        var transformer = Import(ml, onnxPath);

        var tmp = zipPath + ".tmp";
        ml.Model.Save(transformer, emptyData.Schema, tmp);
        File.Move(tmp, zipPath, overwrite: true);
    }
}
