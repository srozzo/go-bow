package main

import (
	"fmt"
	"github.com/srozzo/go-bow"
	"sync"
)

func main() {
	docs := []string{
		"go fast", "fast go", "test case", "code test", "run run run",
	}

	// 🟢 **Concurrent Processing: Raw Frequency Mode**
	fmt.Println("----- Concurrent Processing (Raw Frequency) -----")
	modelRaw := bow.New(bow.Config{
		UseBinaryEncoding: false,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	var wg sync.WaitGroup
	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			_ = modelRaw.AddDocument(d)
		}(doc)
	}
	wg.Wait()

	fmt.Println("Vocabulary:", modelRaw.GetVocab())
	fmt.Println("Raw Document Vectors:")
	for i, vec := range modelRaw.GetRawDocVectors() {
		fmt.Printf("Doc %d: %v\n", i+1, vec)
	}

	// 🟢 **Concurrent Processing: Binary Encoding Mode**
	fmt.Println("\n----- Concurrent Processing (Binary Encoding) -----")
	modelBinary := bow.New(bow.Config{
		UseBinaryEncoding: true,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			_ = modelBinary.AddDocument(d)
		}(doc)
	}
	wg.Wait()

	fmt.Println("Vocabulary:", modelBinary.GetVocab())
	fmt.Println("Binary Document Vectors:")
	for i, vec := range modelBinary.GetRawDocVectors() {
		fmt.Printf("Doc %d: %v\n", i+1, vec)
	}

	// 🟢 **Concurrent Processing: TF-IDF Weighting Mode**
	fmt.Println("\n----- Concurrent Processing (TF-IDF Weighting) -----")
	modelTFIDF := bow.New(bow.Config{
		UseBinaryEncoding: false, // TF-IDF requires frequency-based encoding
		Weighting:         bow.WeightingTFIDF,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			_ = modelTFIDF.AddDocument(d)
		}(doc)
	}
	wg.Wait()

	tfidfVectors, err := modelTFIDF.GetDocVectorsWeighted()
	if err != nil {
		fmt.Println("Error computing TF-IDF:", err)
		return
	}

	fmt.Println("Vocabulary:", modelTFIDF.GetVocab())
	fmt.Println("TF-IDF Weighted Document Vectors:")
	for i, vec := range tfidfVectors {
		fmt.Printf("Doc %d: %v\n", i+1, vec)
	}
}
