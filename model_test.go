// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

import "testing"

func simpleInstances() []TrainingInstance {
	instances := []TrainingInstance{
		{0, FromDenseVector([]float64{1, 1, 1, 0, 0})},
		{0, FromDenseVector([]float64{0, 1, 0, 0, 0})},
		{1, FromDenseVector([]float64{1, 0, 1, 1, 1})},
		{1, FromDenseVector([]float64{0, 0, 0, 1, 1})}}

	return instances
}

func simpleProblem(t *testing.T) *Problem {
	problem := NewProblem()

	for _, instance := range simpleInstances() {
		problem.Add(instance)
	}

	return problem
}

func TestPredict(t *testing.T) {
	problem := simpleProblem(t)

	param := DefaultParameters()

	model, err := TrainModel(param, problem)
	if err != nil {
		t.Error("Could not train model: " + err.Error())
	}

	if model == nil {
		return // We already reported the error.
	}

	check1 := model.Predict(FromDenseVector([]float64{1, 1, 0, 0, 0}))
	if check1 != 0 {
		t.Errorf("Predict(check1) = %f, want 0.0", check1)
	}

	check2 := model.Predict(FromDenseVector([]float64{0, 0, 0, 1, 1}))
	if check2 != 1.0 {
		t.Errorf("Predict(check2) = %f, want 1.0", check2)
	}
}

func TestPredictProbability(t *testing.T) {
	problem := simpleProblem(t)

	param := DefaultParameters()
	param.SolverType = NewL2RLogisticRegressionDefault()

	model, err := TrainModel(param, problem)
	if err != nil {
		t.Error("Could not train model: " + err.Error())
	}

	check1, probs1, err1 := model.PredictProbability(FromDenseVector([]float64{1, 1, 0, 0, 0}))

	if err1 != nil {
		t.Errorf("The model does not support probability estimations")
	}

	if check1 != 0 {
		t.Errorf("Predict(check1) = %f, want 0.0", check1)
	}

	if probs1[0] <= probs1[1] {
		t.Error("p(l0) <= p(l1), want p(l0) > p(l1)")
	}

	check2, probs2, _ := model.PredictProbability(FromDenseVector([]float64{0, 0, 0, 1, 1}))

	if check2 != 1.0 {
		t.Errorf("Predict(check2) = %f, want 1.0", check2)
	}

	if probs2[1] <= probs2[0] {
		t.Error("p(l1) <= p(l0), want p(l1) > p(l0)")
	}
}
