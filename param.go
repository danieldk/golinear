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

func NewL2RLogisticRegression(epsilon float64) SolverType {
	return SolverType{C.L2R_LR, C.double(epsilon)}
}

func NewL2RLogisticRegressionDefault() SolverType {
	return NewL2RLogisticRegression(0.01)
}

func NewL2RL2LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC_DUAL, C.double(epsilon)}
}

func NewL2RL2LossSvcDualDefault() SolverType {
	return NewL2RL2LossSvcDual(0.1)
}

func NewL2RL2LossSvcPrimal(epsilon float64) SolverType {
	return SolverType{C.L2R_L2LOSS_SVC, C.double(epsilon)}
}

func NewL2RL2LossSvcPrimalDefault() SolverType {
	return NewL2RL2LossSvcPrimal(0.01)
}

func NewL2RL1LossSvcDual(epsilon float64) SolverType {
	return SolverType{C.L2R_L1LOSS_SVC_DUAL, C.double(epsilon)}
}

func NewL2RL1LossSvcDualDefault() SolverType {
	return NewL2RL1LossSvcDual(0.1)
}

func NewMCSVMCS(epsilon float64) SolverType {
	return SolverType{C.MCSVM_CS, C.double(epsilon)}
}

func NewMCSVMCSDefault(epsilon float64) SolverType {
	return NewMCSVMCS(0.1)
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
