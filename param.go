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
}

type ClassWeight struct {
	Label int
	Value float64
}

type SolverType struct {
	solverType C.int
	epsilon    C.double
}

// L2-regularized logistic regression (primal).
func NewL2RLogisticRegression(epsilon float64) SolverType {
	return SolverType{C.L2R_LR, C.double(epsilon)}
}

// L2-regularized logistic regression (primal), epsilon = 0.01.
func NewL2RLogisticRegressionDefault() SolverType {
	return NewL2RLogisticRegression(0.01)
}

// L2-regularized L2-loss support vector classification (dual).
func NewL2RL2LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC_DUAL, C.double(epsilon)}
}

// L2-regularized L2-loss support vector classification (dual), epsilon = 0.1.
func NewL2RL2LossSvcDualDefault() SolverType {
	return NewL2RL2LossSvcDual(0.1)
}

// L2-regularized L2-loss support vector classification (primal).
func NewL2RL2LossSvcPrimal(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC, C.double(epsilon)}
}

// L2-regularized L2-loss support vector classification (primal), epsilon = 0.01.
func NewL2RL2LossSvcPrimalDefault() SolverType {
	return NewL2RL2LossSvcPrimal(0.01)
}

// L2-regularized L1-loss support vector classification (dual).
func NewL2RL1LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L1LOSS_SVC_DUAL, C.double(epsilon)}
}

// L2-regularized L1-loss support vector classification (dual), epsilon = 0.1.
func NewL2RL1LossSvcDualDefault() SolverType {
	return NewL2RL1LossSvcDual(0.1)
}

// Support vector classification by Crammer and Singer.
func NewMCSVMCS(epsilon float64) SolverType {
	return SolverType{C.MCSVM_CS, C.double(epsilon)}
}

// Support vector classification by Crammer and Singer, epsilon = 0.1.
func NewMCSVMCSDefault() SolverType {
	return NewMCSVMCS(0.1)
}

// L1-regularized L2-loss support vector classification.
func NewL1RL2LossSvc(epsilon float64) SolverType {
	return SolverType{C.L1R_L2LOSS_SVC, C.double(epsilon)}
}

// L1-regularized L2-loss support vector classification, epsilon = 0.01.
func NewL1RL2LossSvcDefault() SolverType {
	return NewL1RL2LossSvc(0.01)
}

// L1-regularized logistic regression.
func NewL1RLogisticRegression(epsilon float64) SolverType {
	return SolverType{C.L1R_LR, C.double(epsilon)}
}

// L1-regularized logistic regression, epsilon = 0.01.
func NewL1RLogisticRegressionDefault() SolverType {
	return NewL1RLogisticRegression(0.01)
}

// L2-regularized logistic regression (dual) for regression.
func NewL2RLogisticRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_LR_DUAL, C.double(epsilon)}
}

// L2-regularized logistic regression (dual) for regression, epsilon = 0.1.
func NewL2RLogisticRegressionDualDefault() SolverType {
	return NewL2RLogisticRegressionDual(0.1)
}

// L2-regularized L2-loss support vector regression (primal).
func NewL2RL2LossSvRegression(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVR, C.double(epsilon)}
}

// L2-regularized L2-loss support vector regression (primal), epsilon = 0.001.
func NewL2RL2LossSvRegressionDefault(epsilon float64) SolverType {
	return NewL2RL2LossSvRegression(0.001)
}

// L2-regularized L2-loss support vector regression (dual).
func NewL2RL2LossSvRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVR_DUAL, C.double(epsilon)}
}

// L2-regularized L2-loss support vector regression (dual), epsilon = 0.1.
func NewL2RL2LossSvRegressionDualDefault(epsilon float64) SolverType {
	return NewL2RL2LossSvRegressionDual(0.1)
}

// L2-regularized L1-loss support vector regression (dual).
func NewL2RL1LossSvRegressionDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L1LOSS_SVR_DUAL, C.double(epsilon)}
}

// L2-regularized L1-loss support vector regression (dual), epsilon = 0.1.
func NewL2RL1LossSvRegressionDualDefault(epsilon float64) SolverType {
	return NewL2RL1LossSvRegressionDual(0.1)
}

func DefaultParameters() Parameters {
	return Parameters{NewL2RL2LossSvcDualDefault(), 1, nil}
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

	return cParam
}
