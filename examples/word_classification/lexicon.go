package word_classification

import (
	"bufio"
	"strconv"
	"strings"
)

type Dictionary map[string]map[string]uint64

func ReadDictionary(r *bufio.Reader) Dictionary {
	dictionary := make(Dictionary)

	l, e := ReadLn(r)
	for e == nil {
		parts := strings.Split(l, " ")

		word := parts[0]
		dictionary[word] = make(map[string]uint64)

		tagsCounts := parts[1:]

		for i := 0; i < len(tagsCounts); i += 2 {
			freq, err := strconv.ParseUint(tagsCounts[i+1], 10, 64)
			if err != nil {
				panic(err)
			}

			dictionary[word][tagsCounts[i]] = freq
		}

		l, e = ReadLn(r)
	}

	return dictionary
}
