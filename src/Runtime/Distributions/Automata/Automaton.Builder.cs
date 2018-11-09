// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Core.Collections;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Helper class which is used to construct new automaton interactively using the builder pattern.
        /// </summary>
        public class Builder
        {
            /// <summary>
            /// 
            /// </summary>
            private readonly List<StateData> states;

            /// <summary>
            /// 
            /// </summary>
            private readonly List<LinkedTransition> transitions;

            /// <summary>
            /// 
            /// </summary>
            private int numRemovedTransitions = 0;

            /// <summary>
            /// 
            /// </summary>
            public Builder()
            {
                this.states = new List<StateData>();
                this.transitions = new List<LinkedTransition>();
            }

            /// <summary>
            /// 
            /// </summary>
            public int StartStateIndex { get; set; }

            /// <summary>
            /// 
            /// </summary>
            public int StatesCount => this.states.Count;

            /// <summary>
            /// 
            /// </summary>
            public int TransitionsCount => this.transitions.Count - this.numRemovedTransitions;

            /// <summary>
            /// 
            /// </summary>
            /// <param name="index"></param>
            /// <returns></returns>
            public StateBuilder this[int index] => new StateBuilder(this, index);

            /// <summary>
            /// 
            /// </summary>
            public StateBuilder Start => this[this.StartStateIndex];

            /// <summary>
            /// 
            /// </summary>
            public static Builder Zero()
            {
                var builder = new Builder();
                builder.SetToZero();
                return builder;
            }

            /// <summary>
            /// 
            /// </summary>
            public static Builder FromAutomaton(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton)
            {
                var result = new Builder();
                result.AddStates(automaton.States);
                result.StartStateIndex = automaton.startStateIndex;
                return result;
            }

            /// <summary>
            /// 
            /// </summary>
            public static Builder ConstantOn(Weight weight, TSequence sequence)
            {
                var result = Builder.Zero();
                result.Start.AddTransitionsForSequence(sequence).SetEndWeight(weight);
                return result;
            }

            /// <summary>
            /// 
            /// </summary>
            public void SetToZero()
            {
                this.states.Clear();
                this.transitions.Clear();
                this.numRemovedTransitions = 0;
                this.StartStateIndex = 0;
                this.AddState();
            }

            /// <summary>
            /// 
            /// </summary>
            public StateBuilder AddState()
            {
                if (this.states.Count >= maxStateCount)
                {
                    throw new AutomatonTooLargeException(MaxStateCount);
                }

                var index = this.states.Count;
                this.states.Add(
                    new StateData
                    {
                        FirstTransition = -1,
                        LastTransition = -1,
                        EndWeight = Weight.Zero,
                    });
                return new StateBuilder(this, index);
            }

            /// <summary>
            /// 
            /// </summary>
            public void AddStates(int count)
            {
                for (var i = 0; i < count; ++i)
                {
                    AddState();
                }
            }

            /// <summary>
            /// 
            /// </summary>
            public void AddStates(StateCollection states)
            {
                var oldStateCount = this.states.Count;
                foreach (var state in states)
                {
                    var stateBuilder = this.AddState();
                    stateBuilder.SetEndWeight(state.EndWeight);
                    foreach (var transition in state.Transitions)
                    {
                        var updatedTransition = transition;
                        updatedTransition.DestinationStateIndex += oldStateCount;
                        stateBuilder.AddTransition(updatedTransition);
                    }
                }
            }

            /// <summary>
            /// 
            /// </summary>
            public void Append(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton,
                int group = 0,
                bool avoidEpsilonTransitions = true)
            {
                var oldStateCount = this.states.Count;

                foreach (var state in automaton.States)
                {
                    var stateBuilder = this.AddState();
                    stateBuilder.SetEndWeight(state.EndWeight);
                    foreach (var transition in state.Transitions)
                    {
                        var updatedTransition = transition;
                        updatedTransition.DestinationStateIndex += oldStateCount;
                        if (group != 0)
                        {
                            updatedTransition.Group = group;
                        }

                        stateBuilder.AddTransition(updatedTransition);
                    }
                }

                var secondStartState = this[oldStateCount + automaton.startStateIndex];

                if (avoidEpsilonTransitions &&
                    (AllEndStatesHaveNoTransitions() || !automaton.Start.HasIncomingTransitions))
                {
                    // Remove start state of appended automaton and copy all its transitions to previous end states
                    for (var i = 0; i < oldStateCount; ++i)
                    {
                        var endState = this[i];
                        if (!endState.CanEnd)
                        {
                            continue;
                        }

                        for (var iterator = secondStartState.TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            var transition = iterator.Value;

                            if (group != 0)
                            {
                                transition.Group = group;
                            }

                            if (transition.DestinationStateIndex == secondStartState.Index)
                            {
                                transition.DestinationStateIndex = endState.Index;
                            }
                            else
                            {
                                transition.Weight = Weight.Product(transition.Weight, endState.EndWeight);
                            }

                            endState.AddTransition(transition);
                        }

                        endState.SetEndWeight(Weight.Product(endState.EndWeight, secondStartState.EndWeight));
                    }

                    this.RemoveState(secondStartState.Index);
                }
                else
                {
                    // Just connect all end states with start state of appended automaton
                    for (var i = 0; i < oldStateCount; i++)
                    {
                        var state = this[i];
                        if (state.CanEnd)
                        {
                            state.AddEpsilonTransition(state.EndWeight, secondStartState.Index, group);
                            state.SetEndWeight(Weight.Zero);
                        }
                    }
                }

                bool AllEndStatesHaveNoTransitions()
                {
                    for (var i = 0; i < oldStateCount; ++i)
                    {
                        var state = this.states[i];
                        if (state.CanEnd && state.FirstTransition != -1)
                        {
                            return false;
                        }
                    }

                    return true;
                }
            }

            /// <summary>
            /// 
            /// </summary>
            public void RemoveState(int stateIndex)
            {
                // After state is removed, all its transitions will be dead
                for (var iterator = this[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                {
                    iterator.MarkRemoved();
                }

                this.states.RemoveAt(stateIndex);

                for (var i = 0; i < this.states.Count; ++i)
                {
                    for (var iterator = this[i].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        if (transition.DestinationStateIndex > stateIndex)
                        {
                            transition.DestinationStateIndex -= 1;
                            iterator.Value = transition;
                        }
                        else if (transition.DestinationStateIndex == stateIndex)
                        {
                            iterator.MarkRemoved();
                        }
                    }
                }
            }

            /// <summary>
            /// Removes a set of states from the automaton where the set is defined by labels not matching
            /// the <paramref name="removeLabel"/>.
            /// </summary>
            /// <param name="labels">State labels</param>
            /// <param name="removeLabel">Label which marks states which should be deleted</param>
            public int RemoveStates(bool[] labels, bool removeLabel)
            {
                var oldToNewStateIdMapping = new int[this.states.Count];
                var newStateId = 0;
                var deadStateCount = 0;
                for (var stateId = 0; stateId < this.states.Count; ++stateId)
                {
                    if (labels[stateId] != removeLabel)
                    {
                        oldToNewStateIdMapping[stateId] = newStateId++;
                    }
                    else
                    {
                        oldToNewStateIdMapping[stateId] = -1;
                        ++deadStateCount;
                    }
                }

                this.StartStateIndex = oldToNewStateIdMapping[this.StartStateIndex];
                if (this.StartStateIndex == -1)
                {
                    // Cannot reach any end state from the start state => the automaton is zero everywhere
                    this.SetToZero();
                    return deadStateCount;
                }

                for (var i = 0; i < this.states.Count; ++i)
                {
                    var newId = oldToNewStateIdMapping[i];
                    if (newId == -1)
                    {
                        // remove all transitions
                        for (var iterator = this[i].TransitionIterator; iterator.Ok; iterator.Next())
                        {
                            iterator.MarkRemoved();
                        }

                        continue;
                    }

                    Debug.Assert(newId <= i);

                    this.states[newId] = this.states[i];

                    // Remap transitions
                    for (var iterator = this[i].TransitionIterator; iterator.Ok; iterator.Next())
                    {
                        var transition = iterator.Value;
                        transition.DestinationStateIndex = oldToNewStateIdMapping[transition.DestinationStateIndex];
                        if (transition.DestinationStateIndex == -1)
                        {
                            iterator.MarkRemoved();
                        }
                        else
                        {
                            iterator.Value = transition;
                        }
                    }
                }

                this.states.RemoveRange(newStateId, this.states.Count - newStateId);

                return deadStateCount;
            }

            /// <summary>
            /// 
            /// </summary>
            public TThis GetAutomaton()
            {
                var result = new TThis();
                this.ToAutomaton(result);
                return result;
            }

            /// <summary>
            /// 
            /// </summary>
            public void ToAutomaton(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> automaton)
            {
                if (this.StartStateIndex < 0 || this.StartStateIndex >= this.states.Count)
                {
                    throw new InvalidOperationException(
                        $"Built automaton must have a valid start state. StartStateIndex = {this.StartStateIndex}, states.Count = {this.states.Count}");
                }

                var hasEpsilonTransitions = false;
                var resultStates = new StateData[this.states.Count];
                var resultTransitions = new Transition[this.transitions.Count - this.numRemovedTransitions];
                var nextResultTransitionIndex = 0;

                for (var i = 0; i < resultStates.Length; ++i)
                {
                    var state = this.states[i];
                    var transitionIndex = state.FirstTransition;
                    state.FirstTransition = nextResultTransitionIndex;
                    while (transitionIndex != -1)
                    {
                        var linked = this.transitions[transitionIndex];

                        if (!linked.removed)
                        {
                            var transition = linked.transition;
                            Debug.Assert(
                                transition.DestinationStateIndex < resultStates.Length,
                                "Destination indexes must be in valid range");
                            resultTransitions[nextResultTransitionIndex] = transition;
                            ++nextResultTransitionIndex;
                            hasEpsilonTransitions = hasEpsilonTransitions || transition.IsEpsilon;
                        }

                        transitionIndex = linked.next;
                    }

                    state.LastTransition = nextResultTransitionIndex;
                    resultStates[i] = state;
                }

                Debug.Assert(
                    nextResultTransitionIndex == resultTransitions.Length,
                    "number of copied transitions must match result array size");

                automaton.stateCollection = new StateCollection(automaton, resultStates, resultTransitions);
                automaton.startStateIndex = this.StartStateIndex;
                automaton.isEpsilonFree = !hasEpsilonTransitions;
            }

            /// <summary>
            /// 
            /// </summary>
            public struct StateBuilder
            {
                /// <summary>
                /// 
                /// </summary>
                private readonly Builder builder;

                /// <summary>
                /// 
                /// </summary>
                public int Index { get; }

                /// <summary>
                /// 
                /// </summary>
                public bool CanEnd => this.builder.states[this.Index].CanEnd;

                /// <summary>
                /// 
                /// </summary>
                public Weight EndWeight => this.builder.states[this.Index].EndWeight;

                /// <summary>
                /// 
                /// </summary>
                public bool HasTransitions => this.builder[this.Index].TransitionIterator.Ok;

                /// <summary>
                /// 
                /// </summary>
                internal StateBuilder(Builder builder, int index)
                {
                    this.builder = builder;
                    this.Index = index;
                }

                /// <summary>
                /// 
                /// </summary>
                public void SetEndWeight(Weight weight)
                {
                    var state = this.builder.states[this.Index];
                    state.EndWeight = weight;
                    this.builder.states[this.Index] = state;
                }

                /// <summary>
                /// 
                /// </summary>
                public StateBuilder AddTransition(Transition transition)
                {
                    var transitionIndex = this.builder.transitions.Count;
                    this.builder.transitions.Add(
                        new LinkedTransition
                        {
                            transition = transition,
                            next = -1,
                        });
                    var state = this.builder.states[this.Index];

                    if (state.LastTransition != -1)
                    {
                        // update "next" field in old tail
                        var oldTail = this.builder.transitions[state.LastTransition];
                        oldTail.next = transitionIndex;
                        this.builder.transitions[state.LastTransition] = oldTail;
                    }
                    else
                    {
                        state.FirstTransition = transitionIndex;
                    }

                    state.LastTransition = transitionIndex;
                    this.builder.states[this.Index] = state;

                    state.LastTransition = transitionIndex;
                    if (state.FirstTransition == -1)
                    {
                        state.FirstTransition = transitionIndex;
                    }
                    
                    return new StateBuilder(this.builder, transition.DestinationStateIndex);
                }

                /// <summary>
                /// Adds a transition to the current state.
                /// </summary>
                /// <param name="elementDistribution">
                /// The element distribution associated with the transition.
                /// If the value of this parameter is <see langword="null"/>, an epsilon transition will be created.
                /// </param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddTransition(
                    Option<TElementDistribution> elementDistribution,
                    Weight weight,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    if (destinationStateIndex == null)
                    {
                        destinationStateIndex = this.builder.AddState().Index;
                    }

                    return this.AddTransition(
                        new Transition(elementDistribution, weight, destinationStateIndex.Value, group));
                }

                /// <summary>
                /// Adds a transition labeled with a given element to the current state.
                /// </summary>
                /// <param name="element">The element.</param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddTransition(
                    TElement element,
                    Weight weight,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    return this.AddTransition(
                        new TElementDistribution {Point = element}, weight, destinationStateIndex, group);
                }

                /// <summary>
                /// Adds an epsilon transition to the current state.
                /// </summary>
                /// <param name="weight">The transition weight.</param>
                /// <param name="destinationStateIndex">
                /// The destination state of the added transition.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The destination state of the added transition.</returns>
                public StateBuilder AddEpsilonTransition(
                    Weight weight, int? destinationStateIndex = null, int group = 0)
                {
                    return this.AddTransition(Option.None, weight, destinationStateIndex, group);
                }

                /// <summary>
                /// Adds a self-transition labeled with a given element to the current state.
                /// </summary>
                /// <param name="element">The element.</param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The current state.</returns>
                public StateBuilder AddSelfTransition(TElement element, Weight weight, int group = 0)
                {
                    return this.AddTransition(element, weight, this.Index, group);
                }

                /// <summary>
                /// Adds a self-transition to the current state.
                /// </summary>
                /// <param name="elementDistribution">
                /// The element distribution associated with the transition.
                /// If the value of this parameter is <see langword="null"/>, an epsilon transition will be created.
                /// </param>
                /// <param name="weight">The transition weight.</param>
                /// <param name="group">The group of the added transition.</param>
                /// <returns>The current state.</returns>
                public StateBuilder AddSelfTransition(
                    Option<TElementDistribution> elementDistribution, Weight weight, byte group = 0)
                {
                    return this.AddTransition(elementDistribution, weight, this.Index, group);
                }


                /// <summary>
                /// Adds a series of transitions labeled with the elements of a given sequence to the current state,
                /// as well as the intermediate states. All the added transitions have unit weight.
                /// </summary>
                /// <param name="sequence">The sequence.</param>
                /// <param name="destinationStateIndex">
                /// The last state in the transition series.
                /// If the value of this parameter is <see langword="null"/>, a new state will be created.
                /// </param>
                /// <param name="group">The group of the added transitions.</param>
                /// <returns>The last state in the added transition series.</returns>
                public StateBuilder AddTransitionsForSequence(
                    TSequence sequence,
                    int? destinationStateIndex = null,
                    int group = 0)
                {
                    var currentState = this;
                    using (var enumerator = sequence.GetEnumerator())
                    {
                        var moveNext = enumerator.MoveNext();
                        while (moveNext)
                        {
                            var element = enumerator.Current;
                            moveNext = enumerator.MoveNext();
                            currentState = currentState.AddTransition(
                                element, Weight.One, moveNext ? null : destinationStateIndex, group);
                        }
                    }

                    return currentState;
                }

                /// <summary>
                /// 
                /// </summary>
                public TransitionIterator TransitionIterator =>
                    new TransitionIterator(this.builder, this.builder.states[this.Index].FirstTransition);
            }

            /// <summary>
            /// 
            /// </summary>
            public struct TransitionIterator
            {
                private readonly Builder builder;
                private int index;

                public TransitionIterator(Builder builder, int index)
                {
                    this.builder = builder;
                    this.index = index;
                    this.SkipRemoved();
                }

                public Transition Value
                {
                    get => this.builder.transitions[this.index].transition;
                    set
                    {
                        var linked = this.builder.transitions[this.index];
                        linked.transition = value;
                        this.builder.transitions[this.index] = linked;
                    }
                }

                public void MarkRemoved()
                {
                    var linked = this.builder.transitions[this.index];
                    Debug.Assert(!linked.removed, "Trying to delete state twice through iterator");
                    ++this.builder.numRemovedTransitions;
                    linked.removed = true;
                    this.builder.transitions[this.index] = linked;
                }

                public bool Ok => this.index != -1;

                public void Next()
                {
                    this.index = this.builder.transitions[this.index].next;
                    this.SkipRemoved();
                }

                private void SkipRemoved()
                {
                    while (this.index != -1 && this.builder.transitions[this.index].removed)
                    {
                        this.index = this.builder.transitions[this.index].next;
                    }
                }
            }

            /// <summary>
            /// 
            /// </summary>
            private struct LinkedTransition
            {
                public Transition transition;
                public int next;
                public bool removed;
            }
        }
    }
}