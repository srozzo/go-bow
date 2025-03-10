# Go Commands
GOCMD = go
GOTEST = $(GOCMD) test -v $(shell go list ./... | grep -v examples)
GOFMT = gofmt -w
GOLINT = golangci-lint run

# Run all tests excluding examples
test:
	$(GOTEST) -race

# Run concurrency-specific tests separately
test-concurrency:
	$(GOTEST) -run=TestConcurrency -race

# Fuzzing tests
fuzz: fuzz-add-document fuzz-tokenizer fuzz-model

fuzz-add-document:
	$(GOTEST) -fuzz=FuzzTestAddDocument -fuzztime=30s ./...

fuzz-tokenizer:
	$(GOTEST) -fuzz=FuzzTestTokenizer -fuzztime=30s ./...

fuzz-model:
	$(GOTEST) -fuzz=FuzzTestModel -fuzztime=30s ./...

# Lint the project
lint:
	$(GOLINT)

# Format code
format:
	$(GOFMT) .

# Clean up fuzzing test cases and binaries
clean:
	rm -rf testdata/ fuzzcache/ *.out

# Run all checks
check: format lint test

# Ensure all targets are treated as phony
.PHONY: test test-concurrency fuzz fuzz-add-document fuzz-tokenizer fuzz-model lint format clean check