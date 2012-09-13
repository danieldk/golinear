package golinear

/*
#cgo LDFLAGS: -llinear
#include <stddef.h>
#include "wrap.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
)

// Represents a feature and its value. The Index of a feature is used
// to uniquely identify the feature, and should start at 1.
type FeatureValue struct {
	Index int
	Value float64
}

// Sparse feature vector, represented as the list (slice) of non-zero
// features.
type FeatureVector []FeatureValue

type byIndex struct{ FeatureVector }

// Training instance, consisting of the label of the instance and
// its feature vector. In classification, the label is an integer
// indicating the class label. In regression, the label is the
// target value, which can be any real number. The label is not used
// for one-class SVMs.
type TrainingInstance struct {
	Label    float64
	Features FeatureVector
}

// A problem is a set of instances and corresponding labels.
type Problem struct {
	problem *C.problem_t
}

func NewProblem() *Problem {
	cProblem := C.problem_new()
	problem := &Problem{cProblem}

	runtime.SetFinalizer(problem, func(p *Problem) {
		C.problem_free(p.problem)
	})

	return problem
}

// Convert a dense feature vector, represented as a slice of feature
// values to the sparse representation used by this package. The
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
	n := C.nodes_new(C.size_t(len(nodes)))

	for idx, val := range nodes {
		C.nodes_put(n, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	return n
}

func (problem *Problem) Add(trainInst TrainingInstance) error {
	err := verifyFeatureIndices(trainInst.Features)
	if err != nil {
		return err
	}

	features := sortedFeatureVector(trainInst.Features)

	nodes := C.nodes_new(C.size_t(len(features)))

	for idx, val := range features {
		C.nodes_put(nodes, C.size_t(idx), C.int(val.Index), C.double(val.Value))
	}

	C.problem_add_train_inst(problem.problem, nodes, C.double(trainInst.Label))

	return nil
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
			return errors.New(
				fmt.Sprintf("Feature index should be at least one: %d:%f", fv.Index, fv.Value))
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
