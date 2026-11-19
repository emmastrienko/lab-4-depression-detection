using Lab4.Shared.Models;
using Lab4.Shared.Text;
using Microsoft.ML;

namespace Lab4.Train.Onnx;

// Передбачення одними й тими самими прикладами у ML.NET та ONNX
public class OnnxSamplePredictor
{
    private readonly MLContext _ml;
    private readonly ITransformer _mlnetModel;
    private readonly ITransformer _onnxModel;

    public OnnxSamplePredictor(MLContext ml, ITransformer mlnetModel, ITransformer onnxModel)
    {
        _ml = ml;
        _mlnetModel = mlnetModel;
        _onnxModel = onnxModel;
    }

    public IEnumerable<(string Text, TweetPrediction Mlnet, OnnxPrediction Onnx)> PredictSamples(string[] samples)
    {
        var netEngine = _ml.Model.CreatePredictionEngine<TweetInput, TweetPrediction>(_mlnetModel);
        var onnxEngine = _ml.Model.CreatePredictionEngine<TweetInput, OnnxPrediction>(_onnxModel);

        foreach (var s in samples)
        {
            var input = new TweetInput { Text = TextCleaner.Clean(s) };
            yield return (s, netEngine.Predict(input), onnxEngine.Predict(input));
        }
    }
}
