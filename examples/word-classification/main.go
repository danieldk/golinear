package main

import (
	"bufio"
	"fmt"
	"github.com/danieldk/golinear"
	"os"
)

func reverseMapping(mapping map[string]int) map[int]string {
	reverse := make(map[int]string)

	for k, v := range mapping {
		reverse[v] = k
	}

	return reverse
}

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s lexicon\n", os.Args[0])
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Printf("Could not open file: %s\n", os.Args[1])
		os.Exit(1)
	}

	r := bufio.NewReader(f)
	dict := readDictionary(r)

	problem, featureMapping, tagMapping, norm := extractFeatures(dict)

	param := golinear.DefaultParameters()
	//param.Probability = true
	//param.CacheSize = 1024
	//param.Shrinking = true
	//param.Kernel = gosvm.NewRBFKernel(1.0 / float64(len(featureMapping)))

	model, err := golinear.TrainModel(param, problem)
	if err != nil {
		panic(err)
	}

	testPrefix := prefixes("Microsoft", 3)
	features := stringFeatureToFeature(testPrefix, featureMapping, norm)

	class := model.Predict(features)

	numberTagMapping := reverseMapping(tagMapping)

	fmt.Printf("Predicted class: %s\n", numberTagMapping[int(class)])
}
