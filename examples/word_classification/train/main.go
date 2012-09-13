package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/danieldk/golinear"
	"github.com/danieldk/golinear/examples/word_classification"
	"log"
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
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s lexicon modelname\n", os.Args[0])
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Printf("Could not open file: %s\n", os.Args[1])
		os.Exit(1)
	}

	r := bufio.NewReader(f)
	dict := word_classification.ReadDictionary(r)

	problem, metadata := word_classification.ExtractFeatures(dict)

	param := golinear.DefaultParameters()

	model, err := golinear.TrainModel(param, problem)
	if err != nil {
		panic(err)
	}

	modelName := os.Args[2]

	err = model.Save(fmt.Sprintf("%s.model", modelName))
	if err != nil {
		panic(err)
	}

	bMetadata, err := json.Marshal(metadata)
	if err != nil {
		panic(err)
	}

	metadataFile, err := os.OpenFile(fmt.Sprintf("%s.metadata", modelName),
		os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}

	metadataFile.Write(bMetadata)

	metadataFile.Close()

	//testPrefix := prefixes("Microsoft", 3)
	//features := stringFeatureToFeature(testPrefix, featureMapping, norm)

	//class := model.Predict(features)

	//numberTagMapping := reverseMapping(tagMapping)

	//fmt.Printf("Predicted class: %s\n", numberTagMapping[int(class)])
}
