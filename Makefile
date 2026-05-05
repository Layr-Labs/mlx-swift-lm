.PHONY: test test-cb

# Run the full test suite (requires Metal; use xcodebuild so shaders compile).
test:
	xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS' \
		2>&1 | xcpretty || xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS'

# Run only the continuous-batching tests.
test-cb:
	xcodebuild test \
		-scheme mlx-swift-lm-Package \
		-destination 'platform=macOS' \
		-only-testing:MLXLMTests/CBRequestStatusTests \
		-only-testing:MLXLMTests/CBOutputCollectorTests \
		-only-testing:MLXLMTests/CBSamplingTests \
		-only-testing:MLXLMTests/CBRepetitionPenaltyTests \
		-only-testing:MLXLMTests/CBBlockHashTests \
		-only-testing:MLXLMTests/CBPrefixCacheTests \
		-only-testing:MLXLMTests/CBSchedulerTests \
		-only-testing:MLXLMTests/CBEngineCoreLifecycleTests \
		-only-testing:MLXLMTests/CBEngineCoreRequestTests \
		-only-testing:MLXLMTests/CBEngineCoreGenerationTests \
		-only-testing:MLXLMTests/CBEngineCoreThreadSafetyTests \
		-only-testing:MLXLMTests/CBBatchedEngineTests \
		-only-testing:MLXLMTests/CBSSDCacheManagerTests \
		-only-testing:MLXLMTests/CBGenerationBatchShapeTests \
		2>&1 | grep -E "Test Case|Test Suite|SUCCEEDED|FAILED|error:"
