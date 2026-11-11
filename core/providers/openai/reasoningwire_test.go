package openai

import (
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestWireDetailsModeFallsBackOnlyWhenNoDetails checks OpenAI-wire ingress under the
// reasoning_details setting. The wire decoder never synthesizes details, so the text rule
// applies only when the payload carries no reasoning_details, and details the payload does
// carry go out unchanged.
func TestWireDetailsModeFallsBackOnlyWhenNoDetails(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{
			name:    "reasoning alone takes the text rule",
			payload: `{"role":"assistant","reasoning":"step by step"}`,
			want:    `{"reasoning":"step by step"}`,
		},
		{
			name:    "reasoning_content alone takes the text rule",
			payload: `{"role":"assistant","reasoning_content":"step by step"}`,
			want:    `{"reasoning_content":"step by step"}`,
		},
		{
			name:    "details alone are forwarded",
			payload: `{"role":"assistant","reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-D","signature":"SIG-D"}]}`,
			want:    `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-D","signature":"SIG-D"}]}`,
		},
		{
			name:    "details beside text are forwarded and the text is not sent",
			payload: `{"role":"assistant","reasoning_content":"TEXT-T","reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-D","signature":"SIG-D"}]}`,
			want:    `{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"DETAIL-D","signature":"SIG-D"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg OpenAIMessage
			require.NoError(t, sonic.Unmarshal([]byte(tt.payload), &msg))

			converted := ConvertOpenAIMessagesToBifrostMessages([]OpenAIMessage{msg})
			got := reasoningWireOf(t, convertBifrostMessagesToOpenAIMessages(converted, schemas.ReasoningReplayReasoningDetails))
			assert.Equal(t, wantReasoning(t, tt.want), got)
		})
	}
}
