package main

import (
	"fmt"
	"github.com/srozzo/go-bow"
)

func main() {
	docs := []string{
		"go go go fast",
		"test case case",
		"run run run test",
	}

	// 🟢 **Example: Raw Frequency Encoding**
	fmt.Println("----- Raw Frequency Encoding -----")
	modelRaw := bow.New(bow.Config{
		UseBinaryEncoding: false,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	for _, doc := range docs {
		_ = modelRaw.AddDocument(doc)
	}

	fmt.Println("Vocabulary:", modelRaw.GetVocab())
	fmt.Println("Raw Document Vectors:")
	for i, vec := range modelRaw.GetRawDocVectors() {
		fmt.Printf("Doc %d: %v\n", i+1, vec)
	}

	// 🟢 **Example: Binary Encoding**
	fmt.Println("\n----- Binary Encoding -----")
	modelBinary := bow.New(bow.Config{
		UseBinaryEncoding: true,
		Weighting:         bow.WeightingRaw,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	for _, doc := range docs {
		_ = modelBinary.AddDocument(doc)
	}

	fmt.Println("Vocabulary:", modelBinary.GetVocab())
	fmt.Println("Binary Document Vectors:")
	for i, vec := range modelBinary.GetRawDocVectors() {
		fmt.Printf("Doc %d: %v\n", i+1, vec)
	}

	// 🟢 **Example: TF-IDF Weighting**
	fmt.Println("\n----- TF-IDF Weighting -----")
	modelTFIDF := bow.New(bow.Config{
		UseBinaryEncoding: false, // TF-IDF is only valid for frequency-based encoding
		Weighting:         bow.WeightingTFIDF,
		Tokenizer:         bow.DefaultTokenizer{},
	})

	for _, doc := range docs {
		_ = modelTFIDF.AddDocument(doc)
	}

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
