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
	"runtime"
	"unsafe"
)

type mallocFunc func() unsafe.Pointer

// Mallocs, garbage-collects on fail, mallocs, panics on fail.
func tryNew(malloc mallocFunc) unsafe.Pointer {
	p := malloc()
	if p == nil {
		// Garbage-collect and try again.
		runtime.GC()
		p = malloc()
		if p == nil {
			panic("not enough memory")
		}
	}
	return p
}

func newLabels(n C.int) *C.int {
	labels := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.labels_new(n))
	})
	return (*C.int)(labels)
}

func newProbs(model *C.model_t) *C.double {
	probs := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.probs_new(model))
	})
	return (*C.double)(probs)
}

func newDouble(n C.size_t) *C.double {
	p := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.double_new(n))
	})
	return (*C.double)(p)
}

func newParameter() *C.parameter_t {
	param := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.parameter_new())
	})
	return (*C.parameter_t)(param)
}

func newProblem() *C.problem_t {
	problem := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.problem_new())
	})
	return (*C.problem_t)(problem)
}

func newNodes(n C.size_t) *C.feature_node_t {
	nodes := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.nodes_new(n))
	})
	return (*C.feature_node_t)(nodes)
}
