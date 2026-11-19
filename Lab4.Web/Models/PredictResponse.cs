namespace Lab4.Web.Models;

public record PredictResponse(string Text, bool Depression, float Probability, float Score);
