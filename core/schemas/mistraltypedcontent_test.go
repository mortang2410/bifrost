package schemas

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mistral's reasoning preview models stream thought as typed content chunks -
// content: [{"type":"thinking","thinking":[{"type":"text","text":"..."}]}] -
// and sometimes deliver the visible text the same way, instead of using the
// sibling reasoning_content field every other reasoning provider spells. A
// delta that typed content as *string failed to decode, so the gateway dropped
// the whole frame: thinking never reached a client, and a reply whose text
// arrived only as a typed chunk read as an empty stream. The frames below are
// copied from the upstream capture of deepseek-v4-1-flash-1017 through the live
// gateway.
func TestChatStreamDeltaFoldsMistralTypedContentChunks(t *testing.T) {
	t.Run("thinking chunk only", func(t *testing.T) {
		raw := `{"id":"a3b290a9fe81413180ac62df2308055e","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"I multiplied seven by six."}],"closed":true}]},"finish_reason":null}]}`

		var response BifrostChatResponse
		require.NoError(t, json.Unmarshal([]byte(raw), &response))
		require.Len(t, response.Choices, 1)
		delta := response.Choices[0].Delta
		require.NotNil(t, delta, "typed thinking chunk must decode instead of dropping the frame")
		assert.Nil(t, delta.Content, "a thinking-only frame carries no visible text")
		require.NotNil(t, delta.Reasoning, "thinking text must land on the reasoning field")
		assert.Equal(t, "I multiplied seven by six.", *delta.Reasoning)
	})

	t.Run("thinking and text chunks in one frame", func(t *testing.T) {
		raw := `{"id":"830e1f23a708457682ca8df9f8aa93d0","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"seven times six"}]},{"type":"text","text":"42"}]},"finish_reason":null}]}`

		var response BifrostChatResponse
		require.NoError(t, json.Unmarshal([]byte(raw), &response))
		delta := response.Choices[0].Delta
		require.NotNil(t, delta)
		require.NotNil(t, delta.Content, "typed text chunk must become visible content")
		assert.Equal(t, "42", *delta.Content)
		require.NotNil(t, delta.Reasoning)
		assert.Equal(t, "seven times six", *delta.Reasoning)
	})

	t.Run("plain string delta is untouched", func(t *testing.T) {
		raw := `{"choices":[{"index":0,"delta":{"role":"assistant","content":"42"},"finish_reason":null}]}`

		var response BifrostChatResponse
		require.NoError(t, json.Unmarshal([]byte(raw), &response))
		delta := response.Choices[0].Delta
		require.NotNil(t, delta)
		require.NotNil(t, delta.Content)
		assert.Equal(t, "42", *delta.Content)
		assert.Nil(t, delta.Reasoning)
	})

	t.Run("frame with no usable part is left to the caller", func(t *testing.T) {
		raw := `{"delta":{"content":[{"type":"image_url","image_url":{"url":"https://example.test/i.png"}}]}}`
		if folded, ok := foldTypedContentChunks([]byte(raw)); ok {
			t.Fatalf("expected no fold for a single-element array with no text part, got %s", folded)
		}
	})
}
