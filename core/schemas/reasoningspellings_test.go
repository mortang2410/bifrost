package schemas

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Assistant reasoning text can arrive under three spellings in a provider response
// or a request. The shared decoder resolves them as reasoning, then
// reasoning_content, then reasoning_text, where the first non-empty value wins, and
// records the winning spelling in ReasoningSource.
func TestAssistantReasoningSpellingsResolveToReasoning(t *testing.T) {
	tests := []struct {
		name       string
		raw        string
		wantText   string
		wantSource string
	}{
		{
			name:       "reasoning alone",
			raw:        `{"reasoning":"R"}`,
			wantText:   "R",
			wantSource: "reasoning",
		},
		{
			name:       "reasoning_content alone",
			raw:        `{"reasoning_content":"RC"}`,
			wantText:   "RC",
			wantSource: "reasoning_content",
		},
		{
			name:       "reasoning_text alone",
			raw:        `{"reasoning_text":"RT"}`,
			wantText:   "RT",
			wantSource: "reasoning_text",
		},
		{
			name:       "reasoning beats reasoning_content and reasoning_text",
			raw:        `{"reasoning":"R","reasoning_content":"RC","reasoning_text":"RT"}`,
			wantText:   "R",
			wantSource: "reasoning",
		},
		{
			name:       "reasoning_content beats reasoning_text",
			raw:        `{"reasoning_content":"RC","reasoning_text":"RT"}`,
			wantText:   "RC",
			wantSource: "reasoning_content",
		},
		{
			name:       "empty reasoning falls through to reasoning_content",
			raw:        `{"reasoning":"","reasoning_content":"RC"}`,
			wantText:   "RC",
			wantSource: "reasoning_content",
		},
		{
			name:       "empty reasoning and reasoning_content fall through to reasoning_text",
			raw:        `{"reasoning":"","reasoning_content":"","reasoning_text":"RT"}`,
			wantText:   "RT",
			wantSource: "reasoning_text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg ChatAssistantMessage
			require.NoError(t, Unmarshal([]byte(tt.raw), &msg))
			require.NotNil(t, msg.Reasoning)
			assert.Equal(t, tt.wantText, *msg.Reasoning)
			assert.Equal(t, tt.wantSource, msg.ReasoningSource)

			require.Len(t, msg.ReasoningDetails, 1,
				"a reasoning.text entry must be synthesized from the winning text")
			require.NotNil(t, msg.ReasoningDetails[0].Text)
			assert.Equal(t, tt.wantText, *msg.ReasoningDetails[0].Text)
		})
	}
}

// When all four spellings are present, the explicit precedence picks the text and
// the caller's reasoning_details stay exactly as sent. The details must not be
// rebuilt from the winning text, or the caller's signature would be lost.
func TestAssistantAllFourSpellingsKeepDetailsStructured(t *testing.T) {
	raw := `{"reasoning":"R","reasoning_content":"RC","reasoning_text":"RT","reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-MARK","signature":"sig-abc"}]}`

	var msg ChatAssistantMessage
	require.NoError(t, Unmarshal([]byte(raw), &msg))

	require.NotNil(t, msg.Reasoning)
	assert.Equal(t, "R", *msg.Reasoning)
	assert.Equal(t, "reasoning", msg.ReasoningSource)

	require.Len(t, msg.ReasoningDetails, 1)
	require.NotNil(t, msg.ReasoningDetails[0].Text)
	assert.Equal(t, "DETAIL-MARK", *msg.ReasoningDetails[0].Text)
	require.NotNil(t, msg.ReasoningDetails[0].Signature)
	assert.Equal(t, "sig-abc", *msg.ReasoningDetails[0].Signature)
}

// A details-only message has no explicit spelling, so the schema decoder leaves
// Reasoning nil and the source empty. The details reach the caller unchanged.
func TestAssistantDetailsOnlyStaysStructured(t *testing.T) {
	raw := `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-MARK","signature":"sig-abc"}]}`

	var msg ChatAssistantMessage
	require.NoError(t, Unmarshal([]byte(raw), &msg))

	assert.Nil(t, msg.Reasoning, "the decoder must not copy detail text into Reasoning")
	assert.Empty(t, msg.ReasoningSource)

	require.Len(t, msg.ReasoningDetails, 1)
	require.NotNil(t, msg.ReasoningDetails[0].Text)
	assert.Equal(t, "DETAIL-MARK", *msg.ReasoningDetails[0].Text)
	require.NotNil(t, msg.ReasoningDetails[0].Signature)
	assert.Equal(t, "sig-abc", *msg.ReasoningDetails[0].Signature)
}

// ReasoningSource is internal bookkeeping. It must never appear on the wire under
// its Go name or under a snake_case name.
func TestReasoningSourceIsNeverSerialized(t *testing.T) {
	reasoning := "RT"
	encoded, err := json.Marshal(ChatAssistantMessage{
		Reasoning:       &reasoning,
		ReasoningSource: "reasoning_text",
	})
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	assert.NotContains(t, decoded, "ReasoningSource")
	assert.NotContains(t, decoded, "reasoning_source")
	assert.Equal(t, reasoning, decoded["reasoning"])
}
