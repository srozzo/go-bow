// Package bow provides a bag-of-words model for text processing.
package bow

// WeightingScheme defines the available weighting methods.
type WeightingScheme string

const (
	// WeightingRaw represents raw frequency weighting.
	WeightingRaw WeightingScheme = "raw"

	// WeightingTFIDF represents TF-IDF weighting.
	WeightingTFIDF WeightingScheme = "tfidf"
)

// String returns the string representation of the WeightingScheme.
func (w WeightingScheme) String() string {
	return string(w)
}

// Config holds configuration options for the BoW model.
type Config struct {
	// UseBinaryEncoding determines whether the document vectors record only presence (0/1)
	// instead of raw term frequencies.
	UseBinaryEncoding bool

	// Weighting specifies the weighting scheme, either raw frequency or TF-IDF.
	Weighting WeightingScheme

	// Tokenizer defines the text tokenization method. If nil, DefaultTokenizer is used.
	Tokenizer Tokenizer
}

// DefaultConfig returns a default configuration for the BoW model.
func DefaultConfig() Config {
	return Config{
		UseBinaryEncoding: false,
		Weighting:         WeightingRaw,
		Tokenizer:         DefaultTokenizer{},
	}
}
