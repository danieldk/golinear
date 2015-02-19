package golinear

/*
#include <linear.h>
#include "wrap.h"
*/
import "C"

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"runtime"
	"strconv"
)

func Read(reader io.Reader) (*Model, error) {
	model := new(Model)
	model.model = newModel()
	runtime.SetFinalizer(model, finalizeModel)

	scanner := bufio.NewScanner(reader)
	scanner.Split(scanLinear)

header:
	for {
		token, err := scanString(scanner)
		if err != nil {
			return nil, err
		}

		switch token {
		case "solver_type":
			solverType, err := scanString(scanner)
			if err != nil {
				return nil, err
			}

			for idx, solver := range solverTypeTable {
				if solverType == solver {
					model.model.param.solver_type = C.int(idx)
					continue header
				}
			}

			return nil, fmt.Errorf("Unknown solver type: %s", solverType)
		case "nr_class":
			nClasses, err := scanInt(scanner)
			if err != nil {
				return nil, err
			}
			model.model.nr_class = C.int(nClasses)
		case "nr_feature":
			nFeatures, err := scanInt(scanner)
			if err != nil {
				return nil, err
			}
			model.model.nr_feature = C.int(nFeatures)
		case "bias":
			bias, err := scanFloat64(scanner)
			if err != nil {
				return nil, err
			}
			model.model.bias = C.double(bias)
		case "w":
			break header
		case "label":
			nClasses := model.model.nr_class
			labels := newLabels(nClasses)
			model.model.label = labels
			for i := C.int(0); i < nClasses; i++ {
				label, err := scanInt(scanner)
				if err != nil {
					return nil, err
				}
				C.set_int_idx(labels, i, C.int(label))
			}
		default:
			return nil, fmt.Errorf("Unknown text in model file: %s", token)
		}
	}

	nFeatures := model.model.nr_feature
	n := nFeatures
	if model.model.bias >= 0 {
		n++
	}

	nVectors := model.model.nr_class
	if nVectors == 2 && model.model.param.solver_type != C.MCSVM_CS {
		nVectors = 1
	}

	model.model.w = newDouble(C.size_t(nFeatures * nVectors))
	for i := C.int(0); i < n; i++ {
		for j := C.int(0); j < nVectors; j++ {
			weight, err := scanFloat64(scanner)
			if err != nil {
				return nil, err
			}

			C.set_double_idx(model.model.w, i*nVectors+j, C.double(weight))
		}
	}

	return model, nil
}

// Write the model to a writer.
func (model *Model) Write(writer io.Writer) error {
	nFeatures := model.model.nr_feature
	n := nFeatures
	if model.model.bias >= 0 {
		n++
	}

	nVectors := model.model.nr_class
	if nVectors == 2 && model.model.param.solver_type != C.MCSVM_CS {
		nVectors = 1
	}

	fmt.Fprintf(writer, "solver_type %s\n", solverTypeTable[model.model.param.solver_type])
	fmt.Fprintf(writer, "nr_class %d\n", model.model.nr_class)

	if model.model.label != nil {
		fmt.Fprint(writer, "label")

		for _, label := range model.Labels() {
			fmt.Fprintf(writer, " %d", label)
		}

		fmt.Fprintln(writer)
	}

	fmt.Fprintf(writer, "nr_feature %d\n", nFeatures)

	fmt.Fprintf(writer, "bias %.16g\n", model.model.bias)

	fmt.Fprintln(writer, "w")
	for i := C.int(0); i < n; i++ {
		for j := C.int(0); j < nVectors; j++ {
			fmt.Fprintf(writer, "%.16g ", C.get_double_idx(model.model.w, i*nVectors+j))
		}
		fmt.Fprintln(writer)
	}

	return nil
}

// Mostly ScanLines, but we want to split on spaces as
// well, without the full unicode-shebang.
func scanLinear(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}

	if i := bytes.IndexAny(data, " \n"); i >= 0 {
		if consumed, ok := skipExtraWS(data, i); ok || atEOF {
			return consumed, data[0:i], nil
		} else {
			// We need extra data, because the last character of the buffer was
			// whitespace and there can be more...
			return 0, nil, nil
		}
	}
	if atEOF {
		return len(data), data, nil
	}

	return 0, nil, nil
}

func skipExtraWS(data []byte, i int) (int, bool) {
	for i = i + 1; i < len(data); i++ {
		if data[i] != ' ' && data[i] != '\n' {
			return i, true
		}
	}

	return 0, false
}

func scanString(scanner *bufio.Scanner) (string, error) {
	ok := scanner.Scan()
	if !ok && scanner.Err != nil {
		return "", scanner.Err()
	}

	return scanner.Text(), nil
}

func scanInt(scanner *bufio.Scanner) (int, error) {
	token, err := scanString(scanner)
	if err != nil {
		return 0, err
	}

	result, err := strconv.ParseInt(token, 10, 64)
	if err != nil {
		return 0, err
	}

	return int(result), nil
}

func scanFloat64(scanner *bufio.Scanner) (float64, error) {
	token, err := scanString(scanner)
	if err != nil {
		return 0, err
	}

	return strconv.ParseFloat(token, 64)
}

var solverTypeTable = []string{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL",
}
