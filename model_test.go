package golinear

import "testing"

func simpleModel(t *testing.T) *Model {
	problem := NewProblem()
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{1, 1, 1, 0, 0})})
	problem.Add(TrainingInstance{0,
		FromDenseVector([]float64{0, 1, 0, 0, 0})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{1, 0, 1, 1, 1})})
	problem.Add(TrainingInstance{1,
		FromDenseVector([]float64{0, 0, 0, 1, 1})})

	param := DefaultParameters()

	model, err := TrainModel(param, problem)
	if err != nil {
		t.Error("Could not train model: " + err.Error())
	}

	return model
}

func TestPredict(t *testing.T) {
	model := simpleModel(t)

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
