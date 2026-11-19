using Lab4.Shared.Models;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;

namespace Lab4.Train.Onnx;

// Експорт навченої моделі в ONNX та огляд графа
public class OnnxExporter
{
    private readonly MLContext _ml;

    public OnnxExporter(MLContext ml)
    {
        _ml = ml;
    }

    // Експорт в ONNX. Для схеми передаються дані лише з колонкою Text,
    // щоб у ONNX-графі не було зайвого входу Label
    public void Export(ITransformer model, string onnxPath)
    {
        var exportSchemaData = _ml.Data.LoadFromEnumerable(new[] { new TweetInput { Text = "sample" } });
        using (var stream = File.Create(onnxPath))
            _ml.Model.ConvertToOnnx(model, exportSchemaData, stream);
    }

    // Входи/виходи ONNX-графа
    public void PrintGraphSchema(string onnxPath)
    {
        using (var session = new InferenceSession(onnxPath))
        {
            Console.WriteLine("\nВходи ONNX:");
            foreach (var (name, meta) in session.InputMetadata)
                Console.WriteLine($"  {name}: {meta.ElementType.Name} [{string.Join(",", meta.Dimensions)}]");
            Console.WriteLine("Виходи ONNX:");
            foreach (var (name, meta) in session.OutputMetadata)
                Console.WriteLine($"  {name}: {meta.ElementType.Name} [{string.Join(",", meta.Dimensions)}]");
        }
    }
}
