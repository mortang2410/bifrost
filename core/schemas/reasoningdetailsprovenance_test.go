package schemas

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The shared decoder fills ReasoningDetails from Reasoning when the payload carries no
// structured details, and marks that entry synthesized. A payload that carries details
// keeps them unmarked, so outbound replay can tell them apart. The mark is never
// serialized.
func TestChatAssistantMessageMarksSynthesizedDetails(t *testing.T) {
	tests := []struct {
		name            string
		raw             string
		wantDetailText  string
		wantSynthesized bool
	}{
		{
			name:            "reasoning alone is synthesized",
			raw:             `{"reasoning":"R"}`,
			wantDetailText:  "R",
			wantSynthesized: true,
		},
		{
			name:            "reasoning_content alone is synthesized",
			raw:             `{"reasoning_content":"RC"}`,
			wantDetailText:  "RC",
			wantSynthesized: true,
		},
		{
			name:            "payload details are not synthesized",
			raw:             `{"reasoning":"R","reasoning_details":[{"index":0,"type":"reasoning.text","text":"D","signature":"sig"}]}`,
			wantDetailText:  "D",
			wantSynthesized: false,
		},
		{
			name:            "payload details with only a signature are not synthesized",
			raw:             `{"reasoning":"R","reasoning_details":[{"index":0,"type":"reasoning.text","signature":"sig"}]}`,
			wantDetailText:  "",
			wantSynthesized: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg ChatAssistantMessage
			require.NoError(t, json.Unmarshal([]byte(tt.raw), &msg))
			require.Len(t, msg.ReasoningDetails, 1)
			assert.Equal(t, tt.wantSynthesized, msg.ReasoningDetailsSynthesized)
			if tt.wantDetailText != "" {
				require.NotNil(t, msg.ReasoningDetails[0].Text)
				assert.Equal(t, tt.wantDetailText, *msg.ReasoningDetails[0].Text)
			}

			out, err := json.Marshal(msg)
			require.NoError(t, err)
			assert.NotContains(t, string(out), "Synthesized",
				"the provenance mark is internal and must not reach any serialized payload")
		})
	}
}
