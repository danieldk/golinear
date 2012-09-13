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
	err := p.Add(TrainingInstance{0, erronous})
	if err == nil {
		t.Error("Erronous feature index should be rejected")
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
