// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

/*
#cgo CFLAGS: -Wall -Wconversion -O3
#cgo LDFLAGS: -llinear -lstdc++ -lm
#include <stddef.h>
#include "wrap.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"sort"
)

// FeatureValue represents a feature and its value. The Index of a feature
// is used to uniquely identify the feature, and should start at 1.
type FeatureValue struct {
	Index int
	Value float64
}

// A FeatureVector is a sparse feature vector, represented as a
// slice of non-zero features.
type FeatureVector []FeatureValue

type byIndex struct{ FeatureVector }

// A TrainingInstance consists of the label of the instance and
// its feature vector. In classification, the label is an integer
// indicating the class label. In regression, the label is the
// target value, which can be any real number. The label is not used
// for one-class SVMs.
type TrainingInstance struct {
	Label    float64
	Features FeatureVector
}

// A Problem is a set of instances and corresponding labels.
type Problem struct {
	problem *C.problem_t
	insts   []*C.feature_node_t
}

// NewProblem constructs a new problem instance. Problems are used
// to store training instances.
func NewProblem() *Problem {
	cProblem := newProblem()
	problem := &Problem{cProblem, nil}
	runtime.SetFinalizer(problem, finalizeProblem)
	return problem
}

func finalizeProblem(p *Problem) {
	for _, nodes := range p.insts {
		C.nodes_free(nodes)
	}
	p.insts = nil
	C.problem_free(p.problem)
}

// FromDenseVector convert sa dense feature vector, represented as a slice
// of feature values to the sparse representation used by this package. The
// features will be numbered 1..len(denseVector). The following vectors
// will be equal:
//
//     gosvm.FromDenseVector([]float64{0.2, 0.1, 0.3, 0.6})
//     gosvm.FeatureVector{{1, 0.2}, {2, 0.1}, {3, 0.3}, {4, 0.6}}
func FromDenseVector(denseVector []float64) FeatureVector {
	fv := make(FeatureVector, len(denseVector))

	for idx, val := range denseVector {
		fv[idx] = FeatureValue{idx + 1, val}
	}

	return fv
}

func cNodes(nodes []FeatureValue) *C.feature_node_t {
	n := newNodes(C.size_t(len(nodes)))

	for idx, val := range nodes {
		C.nodes_put(n, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	return n
}

// Add adds a training instance to the problem.
func (problem *Problem) Add(trainInst TrainingInstance) error {
	if err := verifyFeatureIndices(trainInst.Features); err != nil {
		return err
	}

	features := sortedFeatureVector(trainInst.Features)

	nodes := newNodes(C.size_t(len(features)))
	problem.insts = append(problem.insts, nodes)

	for idx, val := range features {
		C.nodes_put(nodes, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	C.problem_add_train_inst(problem.problem, nodes, C.double(trainInst.Label))

	return nil
}

// Bias return the bias term.
func (problem *Problem) Bias() float64 {
	return float64(C.problem_bias(problem.problem))
}

// SetBias sets the bias term. Setting this value to non-zero amounts to
// adding an extra feature to each instance with the bias as its value.
func (problem *Problem) SetBias(bias float64) {
	C.set_problem_bias(problem.problem, C.double(bias))
}

// ProblemIterFunc is the function prototype for iteration over problems.
// The function should return 'true' if the iteration should continue or
// 'false' otherwise.
type ProblemIterFunc func(instance *TrainingInstance) bool

// Iterate over the training instances in a problem.
func (problem *Problem) Iterate(fun ProblemIterFunc) {
	for i := 0; i < int(problem.problem.l); i++ {
		label := float64(C.get_double_idx(problem.problem.y, C.int(i)))
		cNodes := C.nodes_vector_get(problem.problem, C.size_t(i))

		fVals := make(FeatureVector, 0)
		var j C.size_t
		for j = 0; C.nodes_get(cNodes, j).index != -1; j++ {
			cNode := C.nodes_get(cNodes, j)
			fVals = append(fVals, FeatureValue{int(cNode.index), float64(cNode.value)})
		}

		if !fun(&TrainingInstance{label, fVals}) {
			break
		}
	}
}

// Helper functions

func sortedFeatureVector(fv FeatureVector) FeatureVector {
	sorted := make(FeatureVector, len(fv))
	copy(sorted, fv)

	sort.Sort(byIndex{sorted})

	return sorted
}

func verifyFeatureIndices(featureVector FeatureVector) error {
	for _, fv := range featureVector {
		if fv.Index < 1 {
			return fmt.Errorf("Feature index should be at least one: %d:%f",
				fv.Index, fv.Value)
		}
	}

	return nil
}

// Interface for sorting of feature vectors by feature index.

func (fv byIndex) Len() int {
	return len(fv.FeatureVector)
}

func (fv byIndex) Swap(i, j int) {
	fv.FeatureVector[i], fv.FeatureVector[j] =
		fv.FeatureVector[j], fv.FeatureVector[i]
}

func (fv byIndex) Less(i, j int) bool {
	return fv.FeatureVector[i].Index < fv.FeatureVector[j].Index
}
