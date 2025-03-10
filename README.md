# go-bow: Bag-of-Words Model in Go

go-bow is a lightweight and efficient Bag-of-Words (BoW) implementation in Go. It provides tokenization, raw frequency encoding, binary encoding, and TF-IDF weighting for document representation.

## Features

- Tokenization with customizable tokenizer support
- Raw frequency encoding
- Binary encoding
- TF-IDF weighting
- Concurrent document processing


## Installation

```sh
go get github.com/srozzo/go-bow
```

## Usage

### Basic Example

```go
package main

import (
	"fmt"
	"log"

	"github.com/srozzo/go-bow"
)

func main() {
	cfg := bow.Config{
		UseBinaryEncoding: false,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	}

	model := bow.New(cfg)

	err := model.AddDocument("go go go fast")
	if err != nil {
		log.Fatalf("Error adding document: %v", err)
	}

	vocab := model.GetVocab()
	fmt.Println("Vocabulary:", vocab)

	vectors := model.GetRawDocVectors()
	fmt.Println("Document Vectors:", vectors)
}
```

### Concurrency Example

```go
package main

import (
	"fmt"
	"log"
	"sync"

	"github.com/srozzo/go-bow"
)

func main() {
	cfg := bow.Config{
		UseBinaryEncoding: false,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	}

	model := bow.New(cfg)

	docs := []string{"go fast", "fast go", "test case", "code test", "run run run"}
	var wg sync.WaitGroup

	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			err := model.AddDocument(d)
			if err != nil {
				log.Printf("Error adding document: %v", err)
			}
		}(doc)
	}
	wg.Wait()

	vocab := model.GetVocab()
	fmt.Println("Vocabulary:", vocab)

	vectors := model.GetRawDocVectors()
	fmt.Println("Document Vectors:", vectors)
}
```

### TF-IDF Example

```go
package main

import (
	"fmt"
	"log"

	"github.com/srozzo/go-bow"
)

func main() {
	cfg := bow.Config{
		UseBinaryEncoding: false,
		Weighting:         bow.WeightingTFIDF,
		Tokenizer:         bow.DefaultTokenizer{},
	}

	model := bow.New(cfg)

	err := model.AddDocuments([]string{
		"go go fast",
		"fast and slow",
		"run fast and free",
		"deep learning models evolve",
	})
	if err != nil {
		log.Fatalf("Error adding documents: %v", err)
	}

	vocab := model.GetVocab()
	fmt.Println("Vocabulary:", vocab)

	tfidfVectors, err := model.GetDocVectorsWeighted()
	if err != nil {
		log.Fatalf("Error computing TF-IDF: %v", err)
	}

	fmt.Println("TF-IDF Vectors:", tfidfVectors)
}
```

## Running Tests

To run the tests:

```sh
make test
```

To run concurrency tests separately:

```sh
make test-concurrency
```

To run fuzz tests:

```sh
make fuzz
```

## License

This project is licensed under the MIT License.