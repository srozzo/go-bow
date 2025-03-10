package bow

import (
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
)

// TestNewBoW verifies that a new BoW instance initializes correctly.
func TestNewBoW(t *testing.T) {
	model := New(DefaultConfig())

	if model == nil {
		t.Fatal("Expected a valid BoW instance, got nil")
	}
	if len(model.vocab) != 0 {
		t.Errorf("Expected empty vocabulary, got %d entries", len(model.vocab))
	}
}

// TestAddDocument uses table-driven tests to verify various cases.
func TestAddDocument(t *testing.T) {
	testCases := []struct {
		name          string
		config        Config
		document      string
		expectedVocab []string
		expectedVec   [][]int
	}{
		{
			name:          "Basic word frequency",
			config:        Config{UseBinaryEncoding: false, Weighting: WeightingRaw},
			document:      "go go go fast",
			expectedVocab: []string{"fast", "go"},
			expectedVec:   [][]int{{1, 3}},
		},
		{
			name:          "Binary mode - word presence only",
			config:        Config{UseBinaryEncoding: true, Weighting: WeightingRaw},
			document:      "go go go fast",
			expectedVocab: []string{"fast", "go"},
			expectedVec:   [][]int{{1, 1}},
		},
		{
			name:          "Handles non-alphanumeric characters",
			config:        Config{UseBinaryEncoding: false, Weighting: WeightingRaw},
			document:      "hello, world! world?",
			expectedVocab: []string{"hello", "world"},
			expectedVec:   [][]int{{1, 2}},
		},
		{
			name:          "Handles empty document",
			config:        Config{UseBinaryEncoding: false, Weighting: WeightingRaw},
			document:      "",
			expectedVocab: []string{},
			expectedVec:   [][]int{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model := New(tc.config)

			err := model.AddDocument(tc.document)
			if err != nil {
				t.Fatalf("Unexpected error adding document: %v", err)
			}

			// Ensure vocabulary is sorted consistently before comparison
			actualVocab := model.GetVocab()
			sort.Strings(actualVocab)

			if !reflect.DeepEqual(actualVocab, tc.expectedVocab) {
				t.Errorf("Vocabulary mismatch.\nGot: %v\nExpected: %v", actualVocab, tc.expectedVocab)
			}

			if !reflect.DeepEqual(model.GetRawDocVectors(), tc.expectedVec) {
				t.Errorf("Document vector mismatch.\nGot: %v\nExpected: %v", model.GetRawDocVectors(), tc.expectedVec)
			}
		})
	}
}

// TestTFIDFMode ensures that TF-IDF weights are computed correctly.
func TestTFIDFMode(t *testing.T) {
	testCases := []struct {
		name       string
		documents  []string
		expectFail bool
	}{
		{
			name: "Basic TF-IDF computation",
			documents: []string{
				"go go fast",
				"fast and slow",
				"run fast and free",
				"deep learning models evolve",
			},
			expectFail: false,
		},
		{
			name:       "Single document TF-IDF",
			documents:  []string{"machine learning AI"},
			expectFail: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model := New(Config{UseBinaryEncoding: false, Weighting: WeightingTFIDF})

			err := model.AddDocuments(tc.documents)
			if err != nil {
				t.Fatalf("Unexpected error adding documents: %v", err)
			}

			tfidfVectors, err := model.GetDocVectorsWeighted()
			if tc.expectFail {
				if err == nil {
					t.Fatalf("Expected an error but got none")
				}
			} else {
				if err != nil {
					t.Fatalf("Unexpected error computing TF-IDF: %v", err)
				}

				hasNonZero := false
				for _, vec := range tfidfVectors {
					for _, value := range vec {
						if value > 0 {
							hasNonZero = true
							break
						}
					}
				}

				if !hasNonZero {
					t.Errorf("All TF-IDF values are zero, which is incorrect: %v", tfidfVectors)
				}
			}
		})
	}
}

// TestConcurrency ensures multiple goroutines can safely add documents.
func TestConcurrency(t *testing.T) {
	model := New(DefaultConfig())
	docs := []string{"go fast", "fast go", "test case", "code test", "run run run"}

	var wg sync.WaitGroup
	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			err := model.AddDocument(d)
			if err != nil {
				t.Errorf("Unexpected error adding document: %v", err)
			}
		}(doc)
	}
	wg.Wait()

	if len(model.GetVocab()) == 0 {
		t.Fatal("Vocabulary should not be empty after concurrent additions")
	}
}

// TestDeadlockStress verifies that high-concurrency access does not lead to deadlocks.
func TestDeadlockStress(t *testing.T) {
	model := New(DefaultConfig())
	docs := make([]string, 500)

	words := []string{"go", "test", "lock", "sync", "parallel", "mutex", "goroutine", "waitgroup", "channel"}
	for i := range docs {
		doc := ""
		for j := 0; j < 5; j++ {
			doc += words[j%len(words)] + " "
		}
		docs[i] = doc
	}

	var wg sync.WaitGroup
	for _, doc := range docs {
		wg.Add(1)
		go func(d string) {
			defer wg.Done()
			err := model.AddDocument(d)
			if err != nil {
				t.Errorf("Unexpected error adding document: %v", err)
			}
		}(doc)
	}
	wg.Wait()
}

// TestAllConfigOptions ensures that all configurations of the BoW model work correctly.
func TestAllConfigOptions(t *testing.T) {
	configs := []Config{
		{UseBinaryEncoding: false, Weighting: WeightingRaw},
		{UseBinaryEncoding: true, Weighting: WeightingRaw},
		{UseBinaryEncoding: false, Weighting: WeightingTFIDF},
		{UseBinaryEncoding: true, Weighting: WeightingTFIDF}, // Should fail
	}

	for _, cfg := range configs {
		t.Run(cfg.Weighting.String()+"_Binary:"+boolToString(cfg.UseBinaryEncoding), func(t *testing.T) {
			model := New(cfg)

			err := model.AddDocuments([]string{"go fast", "run code", "deep learning"})
			if err != nil {
				t.Fatalf("Unexpected error adding documents: %v", err)
			}

			if cfg.Weighting == WeightingTFIDF && cfg.UseBinaryEncoding {
				_, err := model.GetDocVectorsWeighted()
				if err == nil {
					t.Fatalf("Expected an error for TF-IDF with binary encoding, but got none")
				}
			}
		})
	}
}

// boolToString converts a boolean value to string.
func boolToString(value bool) string {
	if value {
		return "True"
	}
	return "False"
}

// FuzzTestTokenizer ensures the tokenizer handles a wide variety of inputs safely.
func FuzzTestTokenizer(f *testing.F) {
	tokenizer := DefaultTokenizer{}

	testCases := []struct {
		name  string
		input string
	}{
		{"Basic Sentence", "Hello, world!"},
		{"Numbers", "12345"},
		{"Repeated Words", "Go! Go! Go!"},
		{"Special Characters", "!@#$%^&*()_+=-{}[]:;<>,.?/"},
		{"Extra Spaces", "This    has   extra spaces"},
		{"Newlines", "newline\nseparated\nwords"},
		{"Tabs", "tab\tseparated\twords"},
		{"Empty Spaces", "     "}, // Should return empty result
		{"Emoji", "Emoji 🚀🔥💡"},
		{"Unicode", "こんにちは 世界"},
		{"NULL Byte", "NULL\x00BYTE"},
		{"Control Characters", "\x01\x02\x03Test\x04\x05\x06"},
	}

	for _, tc := range testCases {
		f.Add(tc.input)
	}

	f.Fuzz(func(t *testing.T, input string) {
		tokens, err := tokenizer.Tokenize(input)
		if err != nil {
			t.Errorf("Tokenizer failed on input %q: %v", input, err)
			return
		}

		// Ensure consistent tokenization for multiple calls
		for i := 0; i < 5; i++ {
			repeatTokens, err := tokenizer.Tokenize(input)
			if err != nil {
				t.Errorf("Tokenizer failed on repeated input %q: %v", input, err)
				return
			}
			if !reflect.DeepEqual(tokens, repeatTokens) {
				t.Errorf("Inconsistent tokenization for input %q.\nExpected: %q\nGot: %q", input, tokens, repeatTokens)
			}
		}
	})
}

// FuzzTestAddDocument ensures `AddDocument()` processes various document types safely.
func FuzzTestAddDocument(f *testing.F) {
	model := New(DefaultConfig())

	testCases := []struct {
		name  string
		input string
	}{
		{"Basic Word Frequency", "go go go fast"},
		{"Simple Case", "test case case"},
		{"Single Word", "run"},
		{"Empty Document", ""},
		{"Non-Alphanumeric", "###!!@@@ $$$%%%^^^"},
		{"Unicode", "こんにちは 世界"},
		{"Long Document", "word " + strings.Repeat("word ", 500)},
	}

	for _, tc := range testCases {
		f.Add(tc.input)
	}

	iterationCount := 0

	f.Fuzz(func(t *testing.T, input string) {
		// Force garbage collection every 100 iterations
		iterationCount++
		if iterationCount%100 == 0 {
			runtime.GC()
		}

		err := model.AddDocument(input)
		if err != nil {
			t.Errorf("Unexpected error adding document: %v", err)
		}
	})
}

// FuzzTestComputeTFIDF ensures TF-IDF computation is robust under various conditions.
func FuzzTestComputeTFIDF(f *testing.F) {
	model := New(Config{UseBinaryEncoding: false, Weighting: WeightingTFIDF})

	testCases := []struct {
		name  string
		input string
	}{
		{"Basic TF-IDF Computation", "go go go fast"},
		{"Multi-Document TF-IDF", "this is a test|test cases vary"},
		{"Single Word Document", "singleword"},
		{"Empty Documents", ""},
		{"Rearranged Words", "random words appear in different places|words random appear places different"},
		{"Numeric Words", "123 456 789|789 456 123"},
	}

	for _, tc := range testCases {
		f.Add(tc.input)
	}

	f.Fuzz(func(t *testing.T, input string) {
		docs := strings.Split(input, "|") // Split into separate documents

		err := model.AddDocuments(docs)
		if err != nil {
			t.Fatalf("Unexpected error adding documents: %v", err)
		}

		tfidfVectors, err := model.GetDocVectorsWeighted()
		if err != nil {
			t.Fatalf("Unexpected error computing TF-IDF: %v", err)
		}

		// Ensure no vector is all zeroes
		hasNonZero := false
		for _, vec := range tfidfVectors {
			for _, value := range vec {
				if value > 0 {
					hasNonZero = true
					break
				}
			}
		}

		if !hasNonZero {
			t.Errorf("All TF-IDF values are zero, which is incorrect: %v", tfidfVectors)
		}
	})
}
