// Package bow provides tokenization utilities for text processing.
package bow

import (
	"strings"
	"unicode"
)

// Tokenizer defines an interface for tokenizing text.
type Tokenizer interface {
	Tokenize(text string) ([]string, error)
}

// DefaultTokenizer is a simple whitespace-based tokenizer.
type DefaultTokenizer struct{}

// Tokenize splits text into lowercase words, removing punctuation.
func (t DefaultTokenizer) Tokenize(text string) ([]string, error) {
	var tokens []string
	var token strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			token.WriteRune(unicode.ToLower(r))
		} else if token.Len() > 0 {
			tokens = append(tokens, token.String())
			token.Reset()
		}
	}

	if token.Len() > 0 {
		tokens = append(tokens, token.String())
	}

	return tokens, nil
}
