// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    public class TrueSkillTests
    {
        [Fact]
        public void RaterDrawMarginPrecisionAndThresholdsModel()
        {
            const double SkillMean = 25.0;
            //const double SkillPrecision = 1;

            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.ReturnCopies = true;
            engine.Compiler.FreeMemory = false;
            //engine.Compiler.UseExistingSourceFiles = true;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.ModelName = "RaterDrawMarginPrecisionAndThresholdsModel";

            var ModelSelector = Variable.Bernoulli(0.5);
            ModelSelector.Name = nameof(ModelSelector);
            //var modelSelectorBlock = Variable.If(ModelSelector);

            // Define the number of games
            var GameCount = Variable.Observed(default(int));
            GameCount.Name = nameof(GameCount);
            var game = new Range(GameCount).Named("game");
            game.AddAttribute(new Sequential());

            // Number of players
            var PlayerCount = Variable.Observed(default(int));
            PlayerCount.Name = nameof(PlayerCount);
            var player = new Range(PlayerCount).Named("player");
            var GamePlayerCount = Variable.Observed(default(int[]), game);
            GamePlayerCount.Name = nameof(GamePlayerCount);
            var gamePlayer = new Range(GamePlayerCount[game]).Named("gamePlayer");

            // Prior skill
            var PlayerSkills = Variable.Array<double>(player);
            PlayerSkills.Name = nameof(PlayerSkills);
            var playerSkillsPriorPrecision = Variable.Array<double>(player).Named("PlayerSkillsPriorPrecision");
            var PlayerSkillsPriorPrecisionPrior = Variable.Observed(default(Gamma[]), player);
            PlayerSkillsPriorPrecisionPrior.Name = nameof(PlayerSkillsPriorPrecisionPrior);
            playerSkillsPriorPrecision[player] = Variable<double>.Random(PlayerSkillsPriorPrecisionPrior[player]);
            PlayerSkills[player] = Variable.GaussianFromMeanAndPrecision(SkillMean, playerSkillsPriorPrecision[player]);
            //PlayerSkills[player] = Variable.GaussianFromMeanAndPrecision(SkillMean, SkillPrecision).ForEach(player);
            bool workaround = false;
            VariableArray<double> PlayerSkills2;
            if (workaround)
            {
                PlayerSkills = Variable.SequentialCopy(PlayerSkills, out PlayerSkills2);
                PlayerSkills.Name = nameof(PlayerSkills) + "Primary";
                PlayerSkills2.Name = nameof(PlayerSkills2);
            }
            else
            {
                PlayerSkills2 = PlayerSkills;
            }

            // Game outcomes
            var PlayerIndices = Variable.Observed(default(int[][]), game, gamePlayer);
            PlayerIndices.Name = nameof(PlayerIndices);
            var PlayerRanks = Variable.Observed(default(int[][]), game, gamePlayer);
            PlayerRanks.Name = nameof(PlayerRanks);

            // Rater count
            var RaterCount = Variable.Observed(default(int));
            RaterCount.Name = nameof(RaterCount);
            var rater = new Range(RaterCount).Named("rater");

            // Rater precision
            var RaterPrecisions = Variable.Array<double>(rater);
            RaterPrecisions.Name = nameof(RaterPrecisions);
            RaterPrecisions[rater] = Variable.Random(Gamma.FromShapeAndRate(10, 1)).ForEach(rater);

            // Raters of ranking
            var RaterIndices = Variable.Observed(default(int[]), game);
            RaterIndices.Name = nameof(RaterIndices);

            // Rater draw margin
            var RaterDrawMargins = Variable.Array<double>(rater);
            RaterDrawMargins.Name = nameof(RaterDrawMargins);
            var RaterDrawMarginsPrior = Variable.Observed(default(Gaussian[]), rater);
            RaterDrawMarginsPrior.Name = nameof(RaterDrawMarginsPrior);
            RaterDrawMargins[rater].SetTo(Variable<double>.Random(RaterDrawMarginsPrior[rater]));
            bool useDrawMargin = false;
            if (useDrawMargin)
            {
                Variable.ConstrainPositive(RaterDrawMargins[rater]);
            }

            // Decks
            using (Variable.ForEach(game))
            {
                var gamePlayerSkills = Variable.Subarray(PlayerSkills, PlayerIndices[game]).Named("GamePlayerSkills");
                var gamePlayerPerformances = Variable.Array<double>(gamePlayer).Named("GamePlayerPerformances");
                gamePlayerPerformances.AddAttribute(new DivideMessages(false));
                gamePlayerPerformances[gamePlayer] = Variable.GaussianFromMeanAndPrecision(
                    gamePlayerSkills[gamePlayer], RaterPrecisions[RaterIndices[game]]);

                var drawMargin = RaterDrawMargins[RaterIndices[game]];
                using (var playerInGame = Variable.ForEach(gamePlayer))
                {
                    using (Variable.If(playerInGame.Index > 0))
                    {
                        var performanceDifference = (gamePlayerPerformances[playerInGame.Index - 1] - gamePlayerPerformances[playerInGame.Index]).Named("PerformanceDifference");

                        if (useDrawMargin)
                        {
                            var isDraw = (PlayerRanks[game][playerInGame.Index] == PlayerRanks[game][playerInGame.Index - 1]).Named("IsDraw");

                            using (Variable.If(isDraw))
                            {
                                Variable.ConstrainBetween(performanceDifference, -drawMargin, drawMargin);
                            }

                            using (Variable.IfNot(isDraw))
                            {
                                Variable.ConstrainTrue(performanceDifference > drawMargin);
                            }
                        }
                        else
                        {
                            Variable.ConstrainTrue(performanceDifference > 0);
                        }
                    }
                }
            }

            // Rater thresholds
            var RaterThresholdCount = Variable.Observed(default(int));
            RaterThresholdCount.Name = nameof(RaterThresholdCount);
            var raterThreshold = new Range(RaterThresholdCount);
            raterThreshold.Name = nameof(raterThreshold);
            var RaterThresholds = Variable.Array(Variable.Array<double>(raterThreshold), rater);
            RaterThresholds.Name = nameof(RaterThresholds);
            var RaterThresholdsPrior = Variable.Observed(default(Gaussian[][]), rater, raterThreshold);
            RaterThresholdsPrior.Name = nameof(RaterThresholdsPrior);
            RaterThresholds[rater][raterThreshold] = Variable<double>.Random(RaterThresholdsPrior[rater][raterThreshold]);

            // Review count
            var ReviewCount = Variable.Observed(default(int));
            ReviewCount.Name = nameof(ReviewCount);
            var review = new Range(ReviewCount).Named("review");
            review.AddAttribute(new Sequential());

            // Review data
            var ReviewRepetitionIndices = Variable.Observed(default(int[]), review);
            ReviewRepetitionIndices.Name = nameof(ReviewRepetitionIndices);
            var ReviewRaterIndices = Variable.Observed(default(int[]), review);
            ReviewRaterIndices.Name = nameof(ReviewRaterIndices);
            var repetitionRatingValue = new Range(RaterThresholdCount - 1);
            repetitionRatingValue.Name = nameof(repetitionRatingValue);
            var ReviewRepetitionRatings = Variable.Observed(default(int[]), review);
            ReviewRepetitionRatings.Name = nameof(ReviewRepetitionRatings);
            ReviewRepetitionRatings.SetValueRange(repetitionRatingValue);

            bool useThresholds = false;
            if (useThresholds)
            {
                // Ordered rater thresholds
                using (Variable.ForEach(rater))
                using (var thresholdForRater = Variable.ForEach(raterThreshold))
                using (Variable.If(thresholdForRater.Index > 0))
                {
                    var thresholdDifference = RaterThresholds[rater][thresholdForRater.Index] - RaterThresholds[rater][thresholdForRater.Index - 1];
                    Variable.ConstrainPositive(thresholdDifference);
                }
            }

            // Reviews
            using (Variable.ForEach(review))
            {
                var repetitionIndex = ReviewRepetitionIndices[review];
                var raterIndex = ReviewRaterIndices[review];
                var repetitionRating = ReviewRepetitionRatings[review];

                var continuousRating = Variable.GaussianFromMeanAndPrecision(
                    PlayerSkills2[repetitionIndex],
                    RaterPrecisions[raterIndex]);

                if (useThresholds)
                {
                    var raterThresholds = Variable.Copy(RaterThresholds[raterIndex]).Named("raterThresholds");
                    using (Variable.Switch(repetitionRating))
                    {
                        // This hack allows indexing the thresholds with the repetitionRatingValue range instead of the raterThreshold range
                        var currentRating = (repetitionRating + 0).Named("CurrentRating");
                        var nextRating = (repetitionRating + 1).Named("NextRating");

                        Variable.ConstrainBetween(continuousRating, raterThresholds[currentRating], raterThresholds[nextRating]);
                    }
                }
                else
                {
                    Variable.ConstrainPositive(continuousRating);
                }
            }
            //modelSelectorBlock.CloseBlock();

            var trainingVariablesToInfer = new IVariable[] { PlayerSkills };
            var TrainingInferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(trainingVariablesToInfer);

            // Data
            TrainingInferenceAlgorithm.SetObservedValue(GameCount.Name, 1);
            TrainingInferenceAlgorithm.SetObservedValue(PlayerCount.Name, 2);
            TrainingInferenceAlgorithm.SetObservedValue(RaterCount.Name, 1);
            TrainingInferenceAlgorithm.SetObservedValue(PlayerSkillsPriorPrecisionPrior.Name, Util.ArrayInit(2, r => Gamma.PointMass(1)));

            TrainingInferenceAlgorithm.SetObservedValue(PlayerIndices.Name, new int[][] { new int[] { 0, 1 } });
            TrainingInferenceAlgorithm.SetObservedValue(GamePlayerCount.Name, new int[] { 2 });
            TrainingInferenceAlgorithm.SetObservedValue(RaterIndices.Name, new int[] { 0 });
            if (useDrawMargin)
            {
                TrainingInferenceAlgorithm.SetObservedValue(RaterDrawMarginsPrior.Name, Util.ArrayInit(1, r => Gaussian.FromMeanAndPrecision(1, 10)));
                TrainingInferenceAlgorithm.SetObservedValue(PlayerRanks.Name, new int[][] { new int[] { 0, 1 } });
            }

            TrainingInferenceAlgorithm.SetObservedValue(ReviewCount.Name, 0);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRaterIndices.Name, new int[0]);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionIndices.Name, new int[0]);
            if (useThresholds)
            {
                TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionRatings.Name, new int[0]);
            }

            // Inference
            for (int i = 1; i <= 1; ++i)
            {
                TrainingInferenceAlgorithm.Execute(i);
            }

            // Posteriors
            var repetitionScoresPosterior = TrainingInferenceAlgorithm.Marginal<Gaussian[]>(PlayerSkills.Name);
            foreach (var score in repetitionScoresPosterior)
            {
                Assert.True(score.GetMean() > SkillMean - 5, $"score = {score}");
            }
        }
    }
}