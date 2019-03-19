// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Attempts to determinize the automaton,
        /// i.e. modify it such that for every state and every element there is at most one transition that allows for that element,
        /// and there are no epsilon transitions.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the determinization attempt was successful and the automaton is now deterministic,
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>See <a href="http://www.cs.nyu.edu/~mohri/pub/hwa.pdf"/> for algorithm details.</remarks>
        public bool TryDeterminize()
        {
            if (this.Data.DeterminizationState != DeterminizationState.Unknown)
            {
                return this.Data.DeterminizationState == DeterminizationState.IsDeterminized;
            }

            int maxStatesBeforeStop = Math.Min(this.States.Count * 3, MaxStateCount);

            this.MakeEpsilonFree(); // Deterministic automata cannot have epsilon-transitions

            if (this.UsesGroups)
            {
                // Determinization will result in lost of group information, which we cannot allow
                this.Data = this.Data.WithDeterminizationState(DeterminizationState.IsNonDeterminizable);
                return false;
            }

            // Weighted state set is a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton..
            // Such pairs correspond to states of the resulting automaton.
            var weightedStateSetQueue = new Queue<Determinization.WeightedStateSet>();
            var weightedStateSetToNewState = new Dictionary<Determinization.WeightedStateSet, int>();
            var builder = new Builder();

            var startWeightedStateSet = new Determinization.WeightedStateSet(this.Start.Index);
            weightedStateSetQueue.Enqueue(startWeightedStateSet);
            weightedStateSetToNewState.Add(startWeightedStateSet, builder.StartStateIndex);
            builder.Start.SetEndWeight(this.Start.EndWeight);

            while (weightedStateSetQueue.Count > 0)
            {
                // Take one unprocessed state of the resulting automaton
                var currentWeightedStateSet = weightedStateSetQueue.Dequeue();
                var currentStateIndex = weightedStateSetToNewState[currentWeightedStateSet];
                var currentState = builder[currentStateIndex];

                // Common special-case: determenistic tansitions from single state.
                // In this case no complicated determinization procedure is needed.
                if (currentWeightedStateSet.Count == 1 &&
                    AllDestinationsAreSame(currentWeightedStateSet[0].Index))
                {
                    Debug.Assert(currentWeightedStateSet[0].Weight == Weight.One);

                    var sourceState = this.States[currentWeightedStateSet[0].Index];
                    foreach (var transition in sourceState.Transitions)
                    {
                        var destinationStates =
                            new Determinization.WeightedStateSet(transition.DestinationStateIndex);
                        var outgoingTransitionInfo =
                            (transition.ElementDistribution.Value, transition.Weight, destinationStates);
                        if (!TryAddTransition(outgoingTransitionInfo, currentState))
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    // Find out what transitions we should add for this state
                    var outgoingTransitionInfos =
                        this.GetOutgoingTransitionsForDeterminization(currentWeightedStateSet);
                    foreach (var outgoingTransitionInfo in outgoingTransitionInfos)
                    {
                        if (!TryAddTransition(outgoingTransitionInfo, currentState))
                        {
                            return false;
                        }
                    }
                }
            }

            var simplification = new Simplification(builder, this.PruneStatesWithLogEndWeightLessThan);
            simplification.MergeParallelTransitions(); // Determinization produces a separate transition for each segment

            this.Data = builder.GetData().WithDeterminizationState(DeterminizationState.IsDeterminized);
            this.PruneStatesWithLogEndWeightLessThan = this.PruneStatesWithLogEndWeightLessThan;
            this.LogValueOverride = this.LogValueOverride;

            return true;

            bool AllDestinationsAreSame(int stateIndex)
            {
                var transitions = this.States[stateIndex].Transitions;
                if (transitions.Count <= 1)
                {
                    return true;
                }

                var destination = transitions[0].DestinationStateIndex;
                for (var i = 1; i < transitions.Count; ++i)
                {
                    if (transitions[i].DestinationStateIndex != destination)
                    {
                        return false;
                    }
                }

                return true;
            }

            bool TryAddTransition(
                (TElementDistribution, Weight, Determinization.WeightedStateSet) outgoingTransitionInfo,
                Builder.StateBuilder currentState)
            {
                var (elementDistribution, weight, destWeightedStateSet) = outgoingTransitionInfo;

                if (!weightedStateSetToNewState.TryGetValue(destWeightedStateSet, out var destinationStateIndex))
                {
                    if (builder.StatesCount == maxStatesBeforeStop)
                    {
                        // Too many states, determinization attempt failed
                        return false;
                    }

                    // Add new state to the result
                    var destinationState = builder.AddState();
                    weightedStateSetToNewState.Add(destWeightedStateSet, destinationState.Index);
                    weightedStateSetQueue.Enqueue(destWeightedStateSet);

                    // Compute its ending weight
                    destinationState.SetEndWeight(Weight.Zero);
                    for (var i = 0; i < destWeightedStateSet.Count; ++i)
                    {
                        var weightedState = destWeightedStateSet[i];
                        var addedWeight = weightedState.Weight * this.States[weightedState.Index].EndWeight;
                        destinationState.SetEndWeight(destinationState.EndWeight + addedWeight);
                    }

                    destinationStateIndex = destinationState.Index;
                }

                // Add transition to the destination state
                currentState.AddTransition(elementDistribution, weight, destinationStateIndex);
                return true;
            }
        }

        /// <summary>
        /// Overridden in the derived classes to compute a set of outgoing transitions
        /// from a given state of the determinization result.
        /// </summary>
        /// <param name="sourceState">The source state of the determinized automaton represented as 
        /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
        /// <returns>
        /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing transitions from <paramref name="sourceState"/>.
        /// The first two elements of a tuple define the element distribution and the weight of a transition.
        /// The third element defines the outgoing state.
        /// </returns>
        protected abstract List<(TElementDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
            Determinization.WeightedStateSet sourceState);

        /// <summary>
        /// Groups together helper classes used for automata determinization.
        /// </summary>
        protected static class Determinization
        {
            public struct WeightedState : IComparable, IComparable<WeightedState>
            {
                public int Index { get; }
                public int WeightHighBits { get; }
                public Weight Weight { get; }

                public WeightedState(int index, Weight weight)
                {
                    this.Index = index;
                    this.WeightHighBits = (int)(BitConverter.DoubleToInt64Bits(weight.LogValue) >> 32);
                    this.Weight = weight;
                }

                public int CompareTo(object obj)
                {
                    return obj is WeightedState that
                        ? this.CompareTo(that)
                        : throw new InvalidOperationException(
                            "WeightedState can be compared only to another WeightedState");
                }

                public int CompareTo(WeightedState that) => Index.CompareTo(that.Index);

                public override int GetHashCode() => (Index ^ WeightHighBits).GetHashCode();
            }

            /// <summary>
            /// Represents a state of the resulting automaton in the power set construction.
            /// It is essentially a set of (stateId, weight) pairs of the source automaton, where each state id is unique.
            /// Supports a quick lookup of the weight by state id.
            /// </summary>
            public struct WeightedStateSet
            {
                /// <summary>
                /// A mapping from state ids to weights.
                /// </summary>
                private readonly ReadOnlyArray<WeightedState> weightedStates;

                private readonly int singleStateIndex;

                public WeightedStateSet(int stateIndex)
                {
                    this.weightedStates = null;
                    this.singleStateIndex = stateIndex;
                }

                public WeightedStateSet(ReadOnlyArray<WeightedState> weightedStates)
                {
                    Debug.Assert(weightedStates.Count > 0);
                    if (weightedStates.Count == 1)
                    {
                        Debug.Assert(weightedStates[0].Weight == Weight.One);
                        this.weightedStates = null;
                        this.singleStateIndex = weightedStates[0].Index;
                    }
                    else
                    {
                        this.weightedStates = weightedStates;
                        this.singleStateIndex = 0; // <- value doesn't matter, but silences the compiler
                    }
                }

                public int Count =>
                    this.weightedStates.IsNull
                        ? 1
                        : this.weightedStates.Count;

                public WeightedState this[int index] =>
                    this.weightedStates.IsNull
                        ? new WeightedState(this.singleStateIndex, Weight.One)
                        : this.weightedStates[index];

                /// <summary>
                /// Checks whether this object is equal to a given one.
                /// </summary>
                /// <param name="obj">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if the objects are equal,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public override bool Equals(object obj) => obj is WeightedStateSet that && this.Equals(that);

                /// <summary>
                /// Checks whether this object is equal to a given one.
                /// </summary>
                /// <param name="that">The object to compare this object with.</param>
                /// <returns>
                /// <see langword="true"/> if the objects are equal,
                /// <see langword="false"/> otherwise.
                /// </returns>
                public bool Equals(WeightedStateSet that)
                {
                    if (this.Count != that.Count)
                    {
                        return false;
                    }

                    // Single-element is treated in custom way for performance reasons
                    if (this.Count == 1)
                    {
                        Debug.Assert(this.weightedStates.IsNull && that.weightedStates.IsNull);
                        return this.singleStateIndex == that.singleStateIndex;
                    }

                    for (var i = 0; i < this.weightedStates.Count; ++i)
                    {
                        var state1 = this.weightedStates[i];
                        var state2 = that.weightedStates[i];
                        if (state1.Index != state2.Index
                            || (state1.WeightHighBits != state2.WeightHighBits
                                && Math.Abs(state1.Weight.LogValue - state2.Weight.LogValue) > 1e-6))
                        {
                            return false;
                        }
                    }

                    return true;
                }

                /// <summary>
                /// Computes the hash code of this instance.
                /// </summary>
                /// <returns>The computed hash code.</returns>
                /// <remarks>Only state ids</remarks>
                public override int GetHashCode()
                {
                    var result = this.weightedStates[0].GetHashCode();
                    for (var i = 1; i < this.weightedStates.Count; ++i)
                    {
                        result = Hash.Combine(result, this.weightedStates[i].GetHashCode());
                    }

                    return result;
                }

                /// <summary>
                /// Returns a string representation of the instance.
                /// </summary>
                /// <returns>A string representation of the instance.</returns>
                public override string ToString() => string.Join(", ", weightedStates);
            }

            public struct WeightedStateSetBuilder
            {
                private List<WeightedState> weightedStates;

                public static WeightedStateSetBuilder Create() =>
                    new WeightedStateSetBuilder()
                    {
                        weightedStates = new List<WeightedState>(1),
                    };

                public void Add(int index, Weight weight) =>
                    this.weightedStates.Add(new WeightedState(index, weight));

                public (WeightedStateSet, Weight) Get()
                {
                    Debug.Assert(weightedStates.Count > 0);

                    var sortedStates = weightedStates.ToArray();
                    if (sortedStates.Length == 1)
                    {
                        var state = sortedStates[0];
                        sortedStates[0] = new WeightedState(state.Index, Weight.One);
                        return (new WeightedStateSet(sortedStates), state.Weight);
                    }
                    else
                    {
                        Array.Sort(sortedStates);

                        var maxWeight = sortedStates[0].Weight;
                        for (var i = 1; i < sortedStates.Length; ++i)
                        {
                            if (sortedStates[i].Weight > maxWeight)
                            {
                                maxWeight = sortedStates[i].Weight;
                            }
                        }

                        var normalizer = Weight.Inverse(maxWeight);

                        for (var i = 0; i < sortedStates.Length; ++i)
                        {
                            var state = sortedStates[i];
                            sortedStates[i] = new WeightedState(state.Index, state.Weight * normalizer);
                        }

                        return (new WeightedStateSet(sortedStates), maxWeight);
                    }
                }
            }
        }
    }
}
