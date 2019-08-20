using System.IO;
using System.Collections.Generic;
using System.Linq;
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using OxyPlot.Wpf;
using OxyPlot;
using OxyPlot.Axes;
using System.Threading;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    public class GPRegression
    {
        // @"..\..\..\Data\ap.txt"
        private string csvPath = @"C:\insurance.csv";
        // 1
        private int randomSeed = 1;

        public void Run()
        {
            var dataset = LoadAISDataset();
            var trainingData = dataset.trainData;
            var validationData = dataset.validationData;

            var trainingInputs = trainingData.Select(tup => Vector.FromArray(new double[1] { System.Math.Round(tup.x, 2) })).ToArray();
            var validationInputs = validationData.Select(tup => Vector.FromArray(new double[1] { System.Math.Round(tup.x, 2) })).ToArray();
            var trainingOutputs = trainingData.Select(tup => System.Math.Round(tup.y, 2)).ToArray();
            var validationOutputs = validationData.Select(tup => System.Math.Round(tup.y, 2)).ToArray();

            Vector[] basis = trainingInputs.Take(7).ToArray();
            //Vector[] basis = new Vector[]
            //{
            //    Vector.FromArray(new double[1] { 0 }),
            //    Vector.FromArray(new double[1] { 0.1 }),
            //    Vector.FromArray(new double[1] { 0.15 }),
            //    Vector.FromArray(new double[1] { 0.2 }),
            //    Vector.FromArray(new double[1] { 0.25 }),
            //    Vector.FromArray(new double[1] { 0.3 }),
            //    Vector.FromArray(new double[1] { 0.5 })
            //};
            //Vector[] basis = new Vector[]
            //{
            //    Vector.FromArray(new double[1] { 0.04 }),
            //    Vector.FromArray(new double[1] { 0.04 }),
            //    Vector.FromArray(new double[1] { 0.1 }),
            //    Vector.FromArray(new double[1] { 0.15 }),
            //    Vector.FromArray(new double[1] { 0.19 }),
            //    Vector.FromArray(new double[1] { 0.36 }),
            //    Vector.FromArray(new double[1] { 0.43 }),


            //};

            //
            //    Vector.FromArray(new double[1] { 0.15 }),
            //    Vector.FromArray(new double[1] { 0.21 }),
            //    Vector.FromArray(new double[1] { 0.23 })

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
            VariableArray<Vector> x = Variable.Observed(trainingInputs).Named("x");
            Range j = x.Range.Named("j");
            VariableArray<double> y = Variable.Observed(trainingOutputs, j).Named("y");
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]);
            y[j] = Variable.GaussianFromMeanAndVariance(score, 0.1);
            block.CloseBlock();

            var kf = new SquaredExponential(-1);
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
            double NNscore = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("{0} evidence = {1}", kf, NNscore.ToString("g4"));

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);
            Console.WriteLine("Predictions on validation set:");
            var plotX = new List<double>();
            var plotY = new List<double>();
            var s1 = new OxyPlot.Series.ScatterSeries
            {
                Title = "Series mrean"
            };
            var s2 = new OxyPlot.Series.ScatterSeries
            {
                Title = "Series points"
            };
            var s3 = new OxyPlot.Series.ScatterSeries
            {
                Title = "Series upper"
            };
            var s4 = new OxyPlot.Series.ScatterSeries
            {
                Title = "Series lower"
            };
            for (int i = 0; i < validationOutputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(validationInputs[i]);
                double postMean = post.GetMean();
                Console.WriteLine("f({0}) = {1}, t = {2}", validationInputs[i], post, validationOutputs[i]);
                plotX.Add(validationInputs[i][0]);
                plotY.Add(postMean);
                s1.Points.Add(new OxyPlot.Series.ScatterPoint(validationInputs[i][0], postMean));
                s2.Points.Add(new OxyPlot.Series.ScatterPoint(validationInputs[i][0], validationOutputs[i]));
                var m = 0.0;
                var p = 0.0;
                post.GetMeanAndPrecision(out m, out p);
                Console.WriteLine(p);
                var stdDev = System.Math.Sqrt(1/ p);
                s3.Points.Add(new OxyPlot.Series.ScatterPoint(validationInputs[i][0], postMean + (2*stdDev)));
                s4.Points.Add(new OxyPlot.Series.ScatterPoint(validationInputs[i][0], postMean - (2 * stdDev)));

            }

            var model = new PlotModel();
            var cakePopularity = Enumerable.Range(1, 5).ToArray();
            var sum = cakePopularity.Sum();
            var barSeries = new OxyPlot.Series.BarSeries
            {
                ItemsSource = cakePopularity
            };

            model.Series.Add(s1);
            model.Series.Add(s2);
            model.Series.Add(s3);
            model.Series.Add(s4);

            //model.Axes.Add(new OxyPlot.Axes.CategoryAxis
            //{
            //    Position = AxisPosition.Left,
            //    Key = "CakeAxis",
            //    ItemsSource = new[]
            //   {
            //   "Apple cake",
            //   "Baumkuchen",
            //   "Bundt Cake",
            //   "Chocolate cake",
            //   "Carrot cake"
            //}
            //});

            Thread thread = new Thread(() => displayPNG(model));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
        }

        private (IEnumerable<(double x, double y)> trainData, IEnumerable<(double x, double y)> validationData) LoadAISDataset()
        {
            var data = new List<(double x, double y)>();

            // Read CSV file
            using (var reader = new StreamReader(this.csvPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    if (values.Length == 2)
                    {
                        data.Add((System.Math.Round(double.Parse(values[0]), 2), System.Math.Round(double.Parse(values[1]))));
                    }
                }
            }

            data = PreprocessData(data).ToList();

            // Split training/validation 80/20
            var rnd = new Random(this.randomSeed);
            var split = (int)System.Math.Floor(data.Count() * 0.8);
            data = data.OrderBy(input => rnd.Next()).ToList();
            var trainData = data.Take(split);
            var validationData = data.Skip(split);
            return (trainData, validationData);
        }

        private IEnumerable<(double x, double y)> PreprocessData(
            IEnumerable<(double x, double y)> data)
        {
            var x = data.Select(tup => tup.x);
            var y = data.Select(tup => tup.y);

            // Shift targets so mean is 0
            var meanY = y.Sum() / y.Count();
            y = y.Select(val => val - meanY);

            // Scale data to lie between 1 and -1
            var absoluteMaxY = y.Select(val => System.Math.Abs(val)).Max();
            y = y.Select(val => val / absoluteMaxY);
            var maxX = x.Max();
            x = x.Select(val => val / maxX);
            return x.Zip(y, (a, b) => (a, b));
        }

        private void displayPNG(PlotModel model)
        {
            var a = Thread.CurrentThread.GetApartmentState();
            var outputToFile = "test-oxyplot-file.png";
            PngExporter.Export(model, outputToFile, 600, 400, OxyColors.White);
        }
    }
}
