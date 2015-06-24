// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

import "testing"

func TestFromDenseVector(t *testing.T) {
	fromDense := FromDenseVector([]float64{0.2, 0.1, 0.3, 0.6})
	check := FeatureVector{{1, 0.2}, {2, 0.1}, {3, 0.3}, {4, 0.6}}

	compareVectors(t, fromDense, check, "fromDense")
}

func TestSortedFeatureVector(t *testing.T) {
	unsorted := FeatureVector{{2, 1}, {1, 0.5}, {3, 1}}
	sorted := sortedFeatureVector(unsorted)
	check := FeatureVector{{1, 0.5}, {2, 1}, {3, 1}}

	compareVectors(t, sorted, check, "sorted")
}

func TestInvalidIndex(t *testing.T) {
	p := NewProblem()
	erronous := FeatureVector{{1, 1}, {2, 0.5}, {0, 1}}
	if err := p.Add(TrainingInstance{0, erronous}); err == nil {
		t.Error("Erronous feature index should be rejected")
	}
}

func TestProblemIterate(t *testing.T) {
	problem := simpleProblem(t)
	instances := simpleInstances()

	idx := 0
	problem.Iterate(func(instance *TrainingInstance) bool {
		check := instances[idx]

		compareVectors(t, instance.Features, check.Features, "iterated")

		if instance.Label != check.Label {
			t.Errorf("label(iterated) = %f, want %f", instance.Label, check.Label)
		}

		idx++

		return true
	})

	// Iteration should be cancelled if the function passed returns 'false'.
	count := 0
	problem.Iterate(func(*TrainingInstance) bool {
		count++
		return false
	})

	if count != 1 {
		t.Error("Problem iteration does not cancel upon the closure's request")
	}
}

func compareVectors(t *testing.T, candidate, check FeatureVector, candidateName string) {
	// Sanity check
	if len(candidate) != len(check) {
		t.Errorf("len(%s) = %d, want %d", candidateName, len(candidate),
			len(check))
		return
	}

	for idx, val := range check {
		if candidate[idx] != val {
			t.Errorf("%s[%d] = (%d, %f), want (%d, %f)", candidateName, idx,
				candidate[idx].Index, candidate[idx].Value, check[idx].Index,
				check[idx].Value)
		}
	}
}
