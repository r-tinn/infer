// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Core.Collections;
    using Microsoft.ML.Probabilistic.Utilities;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a collection of automaton states for use in public APIs
        /// </summary>
        /// <remarks>
        /// Is a thin wrapper around Automaton.stateData. Wraps each <see cref="StateData"/> into <see cref="State"/> on demand.
        /// </remarks>
        public struct StateCollection : IReadOnlyList<State>
        {
            /// <summary>
            /// TODO
            /// </summary>
            internal ReadOnlyArray<StateData> states;

            internal ReadOnlyArray<Transition> transitions;

            /// <summary>
            /// Owner automaton of all states in collection.
            /// </summary>
            private readonly Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner;

            /// <summary>
            /// Initializes instance of <see cref="StateCollection"/>.
            /// </summary>
            internal StateCollection(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis> owner,
                ReadOnlyArray<StateData> states,
                ReadOnlyArray<Transition> transitions)
            {
                this.owner = owner;
                this.states = states;
                this.transitions = transitions;
            }

            #region IReadOnlyList<State> methods

            /// <inheritdoc/>
            public State this[int index] => new State(this.owner, this.states, this.transitions, index);

            /// <inheritdoc/>
            public int Count => this.states.Count;

            /// <summary>
            /// Returns enumerator over all states.
            /// </summary>
            /// <remarks>
            /// This is value-type non-virtual version of enumerator that is used by compiler in foreach loops.
            /// </remarks>
            public StateEnumerator GetEnumerator() => new StateEnumerator(this);

            /// <inheritdoc/>
            IEnumerator<State> IEnumerable<State>.GetEnumerator() => new StateEnumerator(this);

            /// <inheritdoc/>
            IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();

            #endregion

            public void SetTo(StateCollection that)
            {
                this.states = that.states;
                this.transitions = that.transitions;
            }

            public void SwapWith(ref StateCollection that)
            {
                Util.Swap(ref this.states, ref that.states);
                Util.Swap(ref this.transitions, ref that.transitions);
            }

            /// <summary>
            /// Checks if indices assigned to given states and their transitions are consistent with each other.
            /// Throws <see cref="ArgumentException"/> if collection is inconsistent
            /// </summary>
            /// <param name="startStateIndex">The start state to check.</param>
            public void CheckConsistency(int startStateIndex)
            {
                if (startStateIndex < 0 || startStateIndex >= this.Count)
                {
                    throw new ArgumentException("Start state has an invalid index.");
                }

                foreach (var state in this)
                {
                    if (state.Data.FirstTransition < 0 || state.Data.LastTransition > this.transitions.Count)
                    {
                        throw new ArgumentException("Transition indices out of range");
                    }

                    foreach (var transition in state.Transitions)
                    {
                        if (transition.DestinationStateIndex < 0
                            || transition.DestinationStateIndex >= this.Count)
                        {
                            throw new ArgumentException("Transition destination indices must point to a valid state.");
                        }
                    }
                }
            }

            public struct StateEnumerator : IEnumerator<State>
            {
                private readonly StateCollection collection;
                private int index;

                public StateEnumerator(StateCollection collection)
                {
                    this.collection = collection;
                    this.index = -1;
                }

                /// <inheritdoc/>
                public void Dispose()
                {
                }

                /// <inheritdoc/>
                public bool MoveNext()
                {
                    ++this.index;
                    return this.index < this.collection.Count;
                }

                /// <inheritdoc/>
                public State Current => this.collection[this.index];

                /// <inheritdoc/>
                object IEnumerator.Current => this.Current;

                /// <inheritdoc/>
                void IEnumerator.Reset()
                {
                    this.index = -1;
                }
            }
        }
    }
}
