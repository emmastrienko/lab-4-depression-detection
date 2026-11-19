using Lab4.Shared.Models;
using Lab4.Shared.Text;
using Microsoft.ML;

namespace Lab4.Train.Data;

// Завантаження CSV, очищення тексту та перетворення мітки 0/1 -> bool
public class TweetDataLoader
{
    private readonly MLContext _ml;

    public TweetDataLoader(MLContext ml)
    {
        _ml = ml;
    }

    public List<TweetData> Load(string path)
    {
        var raw = _ml.Data.LoadFromTextFile<RawTweet>(
            path, separatorChar: ',', hasHeader: true, allowQuoting: true, trimWhitespace: true);

        return _ml.Data.CreateEnumerable<RawTweet>(raw, reuseRowObject: false)
            .Select(r => new TweetData { Text = TextCleaner.Clean(r.Text), Label = r.Label >= 0.5f })
            .Where(t => !string.IsNullOrWhiteSpace(t.Text))
            .ToList();
    }
}
