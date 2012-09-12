package golinear

/*
#include "wrap.h"
*/
import "C"

type Parameters struct {
	SolverType SolverType
	Cost       C.double
}

type SolverType struct {
	solverType C.int
	epsilon    C.double
}

func NewL2RL2LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC_DUAL, C.double(epsilon)}
}

func NewL2RL2LossSvcDualDefault() SolverType {
	return NewL2RL2LossSvcDual(0.1)
}

func DefaultParameters() Parameters {
	return Parameters{NewL2RL2LossSvcDualDefault(), 1}
}

func toCParameter(param Parameters) *C.parameter_t {
	cParam := C.parameter_new()

	cParam.solver_type = param.SolverType.solverType
	cParam.eps = param.SolverType.epsilon
	cParam.C = param.Cost

	return cParam
}
