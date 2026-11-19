namespace Lab4.Shared.Onnx;

// Назви входів/виходів та форми тензорів ONNX-графа
public static class OnnxNames
{
    public const string Input = "Text";
    public const string Probability = "Probability.output";
    public const string Score = "Score.output";

    public static readonly string[] Inputs = { Input };
    public static readonly string[] Outputs = { Probability, Score };

    public static readonly Dictionary<string, int[]> Shapes = new()
    {
        [Input] = new[] { 1, 1 },
        [Probability] = new[] { 1, 1 },
        [Score] = new[] { 1, 1 }
    };
}
