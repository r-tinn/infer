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
            const double SkillPrecision = 1;

            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.ReturnCopies = true;
            engine.Compiler.FreeMemory = false;
            //engine.Compiler.UseExistingSourceFiles = true;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.ModelName = "RaterDrawMarginPrecisionAndThresholdsModel";

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
            PlayerSkills[player] = Variable.GaussianFromMeanAndPrecision(SkillMean, SkillPrecision).ForEach(player);
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

            // Decks
            using (Variable.ForEach(game))
            {
                var gamePlayerSkills = Variable.Subarray(PlayerSkills, PlayerIndices[game]).Named("GamePlayerSkills");
                var gamePlayerPerformances = Variable.Array<double>(gamePlayer).Named("GamePlayerPerformances");
                gamePlayerPerformances.AddAttribute(new DivideMessages(false));
                gamePlayerPerformances[gamePlayer] = Variable.GaussianFromMeanAndPrecision(
                    gamePlayerSkills[gamePlayer], RaterPrecisions[RaterIndices[game]]);

                using (var playerInGame = Variable.ForEach(gamePlayer))
                {
                    using (Variable.If(playerInGame.Index > 0))
                    {
                        var performanceDifference = (gamePlayerPerformances[playerInGame.Index - 1] - gamePlayerPerformances[playerInGame.Index]).Named("PerformanceDifference");
                        Variable.ConstrainTrue(performanceDifference > 0);
                    }
                }
            }

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

            // Reviews
            using (Variable.ForEach(review))
            {
                var repetitionIndex = ReviewRepetitionIndices[review];
                var raterIndex = ReviewRaterIndices[review];

                var continuousRating = Variable.GaussianFromMeanAndPrecision(
                    PlayerSkills2[repetitionIndex],
                    RaterPrecisions[raterIndex]);

                Variable.ConstrainPositive(continuousRating);
            }

            var trainingVariablesToInfer = new IVariable[] { PlayerSkills };
            var TrainingInferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(trainingVariablesToInfer);

            // Data
            TrainingInferenceAlgorithm.SetObservedValue(GameCount.Name, 1);
            TrainingInferenceAlgorithm.SetObservedValue(PlayerCount.Name, 2);
            TrainingInferenceAlgorithm.SetObservedValue(RaterCount.Name, 1);

            TrainingInferenceAlgorithm.SetObservedValue(PlayerIndices.Name, new int[][] { new int[] { 0, 1 } });
            TrainingInferenceAlgorithm.SetObservedValue(GamePlayerCount.Name, new int[] { 2 });
            TrainingInferenceAlgorithm.SetObservedValue(RaterIndices.Name, new int[] { 0 });

            TrainingInferenceAlgorithm.SetObservedValue(ReviewCount.Name, 0);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRaterIndices.Name, new int[0]);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionIndices.Name, new int[0]);
            //TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionRatings.Name, new int[0]);

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