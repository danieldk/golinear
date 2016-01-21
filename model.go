// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

/*
#include <stdlib.h>
#include "wrap.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// A Model contains the trained model and can be used to predict the
// class of a seen or unseen instance.
type Model struct {
	model *C.model_t
	// Keep a pointer to the problem, since C model depends on it.
	problem *Problem
	// Computing the labels for each classification gets a bit
	// expensive, cache the labels when they are used.
	labelCache []int
}

// Weights extracts the weight vector of a two-class problem.
func (model *Model) Weights() []float64 {
	if model.model.nr_class != 2 {
		panic(fmt.Sprint("not exactly two classes: ", model.model.nr_class))
	}
	n := model.model.nr_feature
	weights := make([]float64, n)
	for i := range weights {
		weights[i] = float64(C.get_double_idx(model.model.w, C.int(i)))
	}
	return weights
}

// Bias extracts the bias of a two-class problem.
func (model *Model) Bias() float64 {
	if model.model.nr_class != 2 {
		panic(fmt.Sprint("not exactly two classes: ", model.model.nr_class))
	}
	// model.nr_feature does not include bias.
	n := model.model.nr_feature
	return float64(C.get_double_idx(model.model.w, n))
}

// WeightsMulti extracts the weight vectors of a multi-class problem.
//
// NOT IMPLEMENTED.
func (model *Model) WeightsMulti() [][]float64 {
	panic("not implemented")
}

// TrainModel trains an SVM using the given parameters and problem.
func TrainModel(param Parameters, problem *Problem) (*Model, error) {
	cParam := toCParameter(param)
	defer func() {
		C.parameter_free(cParam)
		C.destroy_param_wrap(cParam)
		C.free(unsafe.Pointer(cParam))
	}()

	// Check validity of the parameters.
	r := C.check_parameter_wrap(problem.problem, cParam)
	if r != nil {
		msg := C.GoString(r)
		return nil, errors.New(msg)
	}

	cmodel := C.train_wrap(problem.problem, cParam)
	model := &Model{cmodel, problem, nil}
	runtime.SetFinalizer(model, finalizeModel)
	return model, nil
}

// LoadModel loads a previously saved model.
func LoadModel(filename string) (*Model, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	model := &Model{C.load_model_wrap(cFilename), nil, nil}

	if model.model == nil {
		return nil, errors.New("Cannot read model: " + filename)
	}

	runtime.SetFinalizer(model, finalizeModel)

	return model, nil
}

// Labels returns a slice with class labels.
func (model *Model) Labels() []int {
	if model.labelCache != nil {
		labels := make([]int, len(model.labelCache))
		copy(labels, model.labelCache)
		return labels
	}

	nClasses := C.get_nr_class_wrap(model.model)
	cLabels := newLabels(nClasses)
	defer C.free(unsafe.Pointer(cLabels))
	C.get_labels_wrap(model.model, cLabels)

	labels := make([]int, int(nClasses))

	for idx := range labels {
		labels[idx] = int(C.get_int_idx(cLabels, C.int(idx)))
	}

	model.labelCache = make([]int, len(labels))
	copy(model.labelCache, labels)

	return labels
}

// Predict the label of an instance using the given model.
func (model *Model) Predict(nodes []FeatureValue) float64 {
	cn := cNodes(nodes)
	defer C.nodes_free(cn)
	return float64(C.predict_wrap(model.model, cn))
}

// PredictProbability predict the label of an instance, given a model
// with probability information. This method returns the label of the
// predicted class and a map of class probabilities. Probability
// estimates are currently given for logistic regression only. If another
// solver is used, the probability of each class is zero.
func (model *Model) PredictProbability(nodes []FeatureValue) (float64, map[int]float64, error) {
	r, probs, err := model.PredictProbabilitySlice(nodes)
	if err != nil {
		return float64(r), nil, err
	}

	// Store the probabilities in a map
	probMap := make(map[int]float64)
	for idx, label := range model.Labels() {
		probMap[label] = probs[idx]
	}

	return float64(r), probMap, nil
}

// PredictProbabilitySlice predicts the label of an instance, given a
// model with probability information. This method returns the label
// of the predicted class and a slice of class probabilities. Probability
// estimates are currently given for logistic regression only. If another
// solver is used, the probability of each class is zero.
//
// The PredictProbability function is more user-friendly, but has the
// overhead of constructing a map. If you are only interested in the
// classes with the highest probabilities, it may be better to use
// this function in conjunction with Labels().
func (model *Model) PredictProbabilitySlice(nodes []FeatureValue) (float64, []float64, error) {
	// Allocate sparse C feature vector.
	cn := cNodes(nodes)
	defer C.nodes_free(cn)

	probs := make([]float64, len(model.Labels()))
	r := C.predict_probability_wrap(model.model, cn, (*C.double)(unsafe.Pointer(&probs[0])))

	return float64(r), probs, nil
}

// PredictDecisionValues predicts the label of an instance. In contrast
// to Predict, it also returns the per-label decision values.
func (model *Model) PredictDecisionValues(nodes []FeatureValue) (float64, map[int]float64, error) {
	r, values, err := model.PredictDecisionValues(nodes)
	if err != nil {
		return r, values, err
	}

	// Store the decision values in a map
	valuesMap := make(map[int]float64)
	for idx, label := range model.Labels() {
		valuesMap[label] = values[idx]
	}

	return r, valuesMap, nil
}

// PredictDecisionValuesSlice predicts the label of an instance. In
// contrast to Predict, it also returns the per-label decision values.
// The PredictDecisionValues function is more user-friendly, but has
// the overhead of constructing a map. If you are only interested in
// the classes with the highest decision values, it may be better to
// use this function in conjunction with Labels().
func (model *Model) PredictDecisionValuesSlice(nodes []FeatureValue) (float64, []float64, error) {
	// Allocate sparse C feature vector.
	cn := cNodes(nodes)
	defer C.nodes_free(cn)

	labels := model.Labels()

	// Allocate C array for decision values.
	values := make([]float64, len(labels))

	r := C.predict_values_wrap(model.model, cn, (*C.double)(unsafe.Pointer(&values[0])))

	return float64(r), values, nil
}

// Save the model to a file.
func (model *Model) Save(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	result := C.save_model_wrap(model.model, cFilename)

	if result == -1 {
		return errors.New("Could not save model to file: " + filename)
	}

	return nil
}

func finalizeModel(model *Model) {
	C.free_and_destroy_model_wrap(model.model)
	model.problem = nil
}
