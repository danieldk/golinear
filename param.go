// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

package golinear

/*
#include "wrap.h"
*/
import "C"

// Parameters for training a linear model.
type Parameters struct {
	// The type of solver
	SolverType SolverType

	// The cost of constraints violation.
	Cost float64
	// The relative penalty for each class.
	RelCosts []ClassWeight
	// The number of threads to use if liblinear is built with OpenMP support.
	// Default value is 0 and will use all the cores.
	NThreads int
}

// ClassWeight instances are used in the solver parameters to scale
// the constraint violation cost of certain labels.
type ClassWeight struct {
	Label int
	Value float64
}

// A SolverType specifies represents one of the liblinear solvers.
type SolverType struct {
	solverType C.int
	epsilon    C.double
}

// NewL2RLogisticRegression creates an L2-regularized logistic regression
// (primal) solver.
func NewL2RLogisticRegression(epsilon float64) SolverType {
	return SolverType{C.L2R_LR, C.double(epsilon)}
}

// NewL2RLogisticRegressionDefault creates an L2-regularized logistic
// regression (primal) solver, epsilon = 0.01.
func NewL2RLogisticRegressionDefault() SolverType {
	return NewL2RLogisticRegression(0.01)
}

// NewL2RL2LossSvcDual creates an L2-regularized L2-loss support vector
// classification (dual) solver.
func NewL2RL2LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC_DUAL, C.double(epsilon)}
}

// NewL2RL2LossSvcDualDefault creates an L2-regularized L2-loss support
// vector classification (dual) solver, epsilon = 0.1.
func NewL2RL2LossSvcDualDefault() SolverType {
	return NewL2RL2LossSvcDual(0.1)
}

// NewL2RL2LossSvcPrimal creates an L2-regularized L2-loss support vector
// classification (primal) solver.
func NewL2RL2LossSvcPrimal(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC, C.double(epsilon)}
}

// NewL2RL2LossSvcPrimalDefault creates an L2-regularized L2-loss support
// vector classification (primal) solver, epsilon = 0.01.
func NewL2RL2LossSvcPrimalDefault() SolverType {
	return NewL2RL2LossSvcPrimal(0.01)
}

// NewL2RL1LossSvcDual creates an L2-regularized L1-loss support vector
// classification (dual) solver.
func NewL2RL1LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L1LOSS_SVC_DUAL, C.double(epsilon)}
}

// NewL2RL1LossSvcDualDefault creates an L2-regularized L1-loss support
// vector classification (dual) solver, epsilon = 0.1.
func NewL2RL1LossSvcDualDefault() SolverType {
	return NewL2RL1LossSvcDual(0.1)
}

// NewMCSVMCS creates a Support vector classification solver
// (Crammer and Singer).
func NewMCSVMCS(epsilon float64) SolverType {
	return SolverType{C.MCSVM_CS, C.double(epsilon)}
}

// NewMCSVMCSDefault creates a Support vector classification solver
// (Crammer and Singer), epsilon = 0.1.
func NewMCSVMCSDefault() SolverType {
	return NewMCSVMCS(0.1)
}

// NewL1RL2LossSvc creates an L1-regularized L2-loss support vector
// classification solver.
func NewL1RL2LossSvc(epsilon float64) SolverType {
	return SolverType{C.L1R_L2LOSS_SVC, C.double(epsilon)}
}

// NewL1RL2LossSvcDefault creates an L1-regularized L2-loss support
// vector classification solver, epsilon = 0.01.
func NewL1RL2LossSvcDefault() SolverType {
	return NewL1RL2LossSvc(0.01)
}

// NewL1RLogisticRegression creates an L1-regularized logistic
// regression solver.
func NewL1RLogisticRegression(epsilon float64) SolverType {
	return SolverType{C.L1R_LR, C.double(epsilon)}
}

// NewL1RLogisticRegressionDefault creates an L1-regularized logistic
// regression solver, epsilon = 0.01.
func NewL1RLogisticRegressionDefault() SolverType {
	return NewL1RLogisticRegression(0.01)
}

// NewL2RLogisticRegressionDual creates an L2-regularized logistic
// regression (dual) for regression solver.
func NewL2RLogisticRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_LR_DUAL, C.double(epsilon)}
}

// NewL2RLogisticRegressionDualDefault creates an L2-regularized logistic
// regression (dual) for regression solver, epsilon = 0.1.
func NewL2RLogisticRegressionDualDefault() SolverType {
	return NewL2RLogisticRegressionDual(0.1)
}

// NewL2RL2LossSvRegression creates an L2-regularized L2-loss support vector
// regression (primal) solver.
func NewL2RL2LossSvRegression(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVR, C.double(epsilon)}
}

// NewL2RL2LossSvRegressionDefault creates an L2-regularized L2-loss support
// vector regression (primal) solver, epsilon = 0.001.
func NewL2RL2LossSvRegressionDefault() SolverType {
	return NewL2RL2LossSvRegression(0.001)
}

// NewL2RL2LossSvRegressionDual creates an L2-regularized L2-loss support
// vector regression (dual) solver.
func NewL2RL2LossSvRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVR_DUAL, C.double(epsilon)}
}

// NewL2RL2LossSvRegressionDualDefault creates an L2-regularized L2-loss
// support vector regression (dual) solver, epsilon = 0.1.
func NewL2RL2LossSvRegressionDualDefault() SolverType {
	return NewL2RL2LossSvRegressionDual(0.1)
}

// NewL2RL1LossSvRegressionDual creates an L2-regularized L1-loss support
// vector regression solver (dual).
func NewL2RL1LossSvRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L1LOSS_SVR_DUAL, C.double(epsilon)}
}

// NewL2RL1LossSvRegressionDualDefault creates an L2-regularized L1-loss
// support vector regression (dual) solver, epsilon = 0.1.
func NewL2RL1LossSvRegressionDualDefault() SolverType {
	return NewL2RL1LossSvRegressionDual(0.1)
}

// DefaultParameters returns a set of reasonable default parameters:
// L2-regularized L2-loss spport vector classification (dual) and a
// constraint violation cost of 1.
func DefaultParameters() Parameters {
	return Parameters{NewL2RL2LossSvcDualDefault(), 1, nil, 0}
}

func toCParameter(param Parameters) *C.parameter_t {
	cParam := newParameter()

	cParam.solver_type = param.SolverType.solverType
	cParam.eps = param.SolverType.epsilon
	cParam.C = C.double(param.Cost)

	// Copy relative costs into C structure.
	n := len(param.RelCosts)
	if n > 0 {
		cParam.nr_weight = C.int(n)
		cParam.weight_label = newLabels(C.int(n))
		cParam.weight = newDouble(C.size_t(n))
		for i, weight := range param.RelCosts {
			C.set_int_idx(cParam.weight_label, C.int(i), C.int(weight.Label))
			C.set_double_idx(cParam.weight, C.int(i), C.double(weight.Value))
		}
	}

	// Set the number of threads to use by OpenMP.
	C.parameter_set_nthreads(cParam, C.int(param.NThreads))

	return cParam
}
