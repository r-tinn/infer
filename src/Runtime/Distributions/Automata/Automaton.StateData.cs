// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Serialization;

    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {
        /// <summary>
        /// Represents a state of an automaton that is stored in the Automaton.states. This is an internal representation
        /// of the state. <see cref="State"/> struct should be used in public APIs.
        /// </summary>
        [Serializable]
        [DataContract]
        public struct StateData
        {
            [DataMember]
            public int FirstTransition { get; internal set; }

            /// <summary>
            /// TODO
            /// </summary>
            [DataMember]
            public int LastTransition { get; internal set; }

            [DataMember]
            public Weight EndWeight { get; internal set; }

            /// <summary>
            /// Initializes a new instance of the <see cref="StateData"/> struct.
            /// </summary>
            [Construction("FirstTransition", "LastTransition", "EndWeight")]
            public StateData(int firstTransition, int lastTransition, Weight endWeight)
            {
                this.FirstTransition = firstTransition;
                this.LastTransition = lastTransition;
                this.EndWeight = endWeight;
            }

            /// <summary>
            /// Gets a value indicating whether the ending weight of this state is greater than zero.
            /// </summary>
            internal bool CanEnd => !this.EndWeight.IsZero;
        }
    }
}
