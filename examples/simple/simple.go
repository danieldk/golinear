package main

import (
	"fmt"
	"github.com/danieldk/golinear"
	"log"
)

func main() {
	problem := golinear.NewProblem()
	problem.Add(golinear.TrainingInstance{0, golinear.FromDenseVector([]float64{1, 1, 1, 0, 0})})
	problem.Add(golinear.TrainingInstance{1, golinear.FromDenseVector([]float64{1, 0, 1, 1, 1})})

	param := golinear.DefaultParameters()
	model, err := golinear.TrainModel(param, problem)
	if err != nil {
		log.Fatal("Could not train the model: " + err.Error())
	}

	label := model.Predict(golinear.FromDenseVector([]float64{1, 1, 0, 0, 0}))

	fmt.Printf("Predicted label: %f\n", label)
}
