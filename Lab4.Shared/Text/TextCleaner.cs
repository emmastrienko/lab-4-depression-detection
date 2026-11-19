using System.Text.RegularExpressions;

namespace Lab4.Shared.Text;

public static class TextCleaner
{
    // Очищення виконується до моделі (і в консолі, і у веб-додатку однаково),
    // бо нормалізація всередині FeaturizeText не експортується в ONNX
    public static string Clean(string text)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;
        text = text.ToLowerInvariant();
        text = Regex.Replace(text, @"https?://\S+|www\.\S+", " ");
        text = Regex.Replace(text, @"@\w+", " ");
        text = Regex.Replace(text, @"&\w+;", " ");
        text = Regex.Replace(text, @"[^a-z]+", " ");
        return Regex.Replace(text, @"\s+", " ").Trim();
    }
}
