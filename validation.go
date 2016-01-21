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
	"unsafe"
)

// CrossValidation separates the problem in folds. Each fold is sequentially
// evaluated using the model trained with the remaining folds. The slice that
// is returned contains the predicted instance classes.
func CrossValidation(problem *Problem, param Parameters, nFolds uint) ([]float64, error) {
	cParam := toCParameter(param)
	defer func() {
		C.destroy_param_wrap(cParam)
		C.free(unsafe.Pointer(cParam))
	}()

	r := C.check_parameter_wrap(problem.problem, cParam)
	if r != nil {
		msg := C.GoString(r)
		return nil, errors.New(msg)
	}

	nInstances := uint(problem.problem.l)
	target := newDouble(C.size_t(nInstances))
	defer C.free(unsafe.Pointer(target))

	C.cross_validation_wrap(problem.problem, cParam, C.int(nFolds), target)

	classifications := make([]float64, nInstances)
	for idx := range classifications {
		classifications[idx] = float64(C.get_double_idx(target, C.int(idx)))
	}

	return classifications, nil
}
