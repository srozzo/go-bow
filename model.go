package bow

import (
	"sort"
	"sync"
)

// BoW represents the bag-of-words model.
type BoW struct {
	cfg   Config
	vocab map[string]int // Maps words to indices.
	docs  [][]int        // Document vectors.

	vocabLock sync.RWMutex
	docLock   sync.RWMutex
}

// New creates a new BoW model with the given configuration.
func New(cfg Config) *BoW {
	if cfg.Tokenizer == nil {
		cfg.Tokenizer = DefaultTokenizer{}
	}

	return &BoW{
		cfg:   cfg,
		vocab: make(map[string]int),
		docs:  [][]int{},
	}
}

// AddDocument tokenizes the document, updates the vocabulary, and appends the document vector to the model.
func (bow *BoW) AddDocument(doc string) error {
	words, err := bow.cfg.Tokenizer.Tokenize(doc)
	if err != nil {
		return err
	}

	if len(words) == 0 { // Handle empty document
		return nil
	}

	wordFreq := make(map[string]int)
	for _, word := range words {
		wordFreq[word]++
	}

	// Acquire write lock before modifying vocab
	bow.vocabLock.Lock()
	newWords := []string{}
	for word := range wordFreq {
		if _, exists := bow.vocab[word]; !exists {
			newWords = append(newWords, word)
		}
	}
	sort.Strings(newWords)
	for _, word := range newWords {
		bow.vocab[word] = len(bow.vocab)
	}
	bow.vocabLock.Unlock()

	// Read lock for vocab access
	bow.vocabLock.RLock()
	vec := make([]int, len(bow.vocab))
	for word, freq := range wordFreq {
		if idx, exists := bow.vocab[word]; exists {
			if bow.cfg.UseBinaryEncoding {
				vec[idx] = 1
			} else {
				vec[idx] = freq
			}
		}
	}
	bow.vocabLock.RUnlock()

	// Write lock for modifying docs list
	bow.docLock.Lock()
	bow.docs = append(bow.docs, vec)
	bow.docLock.Unlock()

	return nil
}

// AddDocuments processes multiple documents in a batch.
func (bow *BoW) AddDocuments(docs []string) error {
	for _, doc := range docs {
		if err := bow.AddDocument(doc); err != nil {
			return err
		}
	}
	return nil
}

// GetVocab returns the vocabulary as a sorted list of words.
func (bow *BoW) GetVocab() []string {
	bow.vocabLock.RLock()
	defer bow.vocabLock.RUnlock()

	vocabList := make([]string, len(bow.vocab))
	for word, idx := range bow.vocab {
		vocabList[idx] = word
	}
	return vocabList
}

// GetRawDocVectors returns a copy of the raw document vectors.
func (bow *BoW) GetRawDocVectors() [][]int {
	bow.docLock.RLock()
	defer bow.docLock.RUnlock()

	docsCopy := make([][]int, len(bow.docs))
	for i, vec := range bow.docs {
		temp := make([]int, len(vec))
		copy(temp, vec)
		docsCopy[i] = temp
	}
	return docsCopy
}
