// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

import "testing"

func tenInstanceProblem(t *testing.T) *Problem {
	problem := NewProblem()
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 1, 0, 0})})
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 1, 0, 0})})
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 0, 0, 0})})
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 0, 0, 0})})
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 0, 0, 0})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 1, 1, 1})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 1, 1, 1})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 0, 1, 1})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 0, 1, 1})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 0, 1, 1})})

	return problem
}

func TestCrossValidation(t *testing.T) {
	problem := tenInstanceProblem(t)
	param := DefaultParameters()

	results, err := CrossValidation(problem, param, 10)
	if err != nil {
		t.Errorf("Could not train model: %s", err.Error())
	}

	correctResults := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
	for idx, class := range correctResults {
		if results[idx] != class {
			t.Errorf("class(%d) = %f, want class(%d) = %f", idx, results[idx], idx, class)
		}
	}
}
