// Package bow provides TF-IDF computations for document vectors.
package bow

import (
	"fmt"
	"math"
)

// ComputeTFIDF computes the TF-IDF weighted vectors based on the raw frequency vectors.
func (bow *BoW) ComputeTFIDF() [][]float64 {
	raw := bow.GetRawDocVectors()
	totalDocs := float64(len(raw))

	bow.vocabLock.RLock()
	vocabSize := len(bow.vocab)
	bow.vocabLock.RUnlock()

	if totalDocs == 0 {
		return [][]float64{}
	}

	// Compute document frequency for each term.
	docFreq := make([]int, vocabSize)
	for _, vec := range raw {
		for j, tf := range vec {
			if tf > 0 {
				docFreq[j]++
			}
		}
	}

	// Compute TF-IDF for each document.
	tfidfMatrix := make([][]float64, len(raw))
	for i, vec := range raw {
		tfidfVec := make([]float64, vocabSize)
		for j, tf := range vec {
			if tf > 0 {
				// Prevent zero IDF by applying smoothing
				idf := math.Log((totalDocs+1)/(float64(docFreq[j])+1)) + 1
				tfidfVec[j] = float64(tf) * idf
			}
		}
		tfidfMatrix[i] = tfidfVec
	}
	return tfidfMatrix
}

// GetDocVectorsWeighted returns document vectors with the configured weighting scheme.
func (bow *BoW) GetDocVectorsWeighted() ([][]float64, error) {
	if bow.cfg.Weighting == WeightingTFIDF {
		if bow.cfg.UseBinaryEncoding {
			return nil, fmt.Errorf("TF-IDF weighting is not supported in binary mode")
		}
		return bow.ComputeTFIDF(), nil
	}

	raw := bow.GetRawDocVectors()
	result := make([][]float64, len(raw))
	for i, vec := range raw {
		floatVec := make([]float64, len(vec))
		for j, val := range vec {
			floatVec[j] = float64(val)
		}
		result[i] = floatVec
	}
	return result, nil
}
