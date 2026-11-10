package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Mistral rejects the sibling reasoning spellings with 422 extra_forbidden: its
// message schema is closed (additionalProperties: false), and its own replay
// form is a "thinking" content chunk. A replayed assistant turn therefore must
// not carry reasoning_content, reasoning, reasoning_text or reasoning_details,
// or every multi-turn request on a Mistral route fails.
func TestToOpenAIChatRequest_MistralDropsReplayedReasoning(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(nil)
	defer cancel()

	userContent := "What is 7*6?"
	answer := "42"
	reasoning := "seven times six"
	reasoningText := reasoning

	buildRequest := func(provider schemas.ModelProvider) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{
			Provider: provider,
			Model:    "deepseek-v4-1-flash-1017",
			Input: []schemas.ChatMessage{
				{
					Role:    schemas.ChatMessageRoleUser,
					Content: &schemas.ChatMessageContent{ContentStr: &userContent},
				},
				{
					Role:    schemas.ChatMessageRoleAssistant,
					Content: &schemas.ChatMessageContent{ContentStr: &answer},
					ChatAssistantMessage: &schemas.ChatAssistantMessage{
						Reasoning: &reasoning,
						ReasoningDetails: []schemas.ChatReasoningDetails{
							{Index: 0, Type: schemas.BifrostReasoningDetailsTypeText, Text: &reasoningText},
						},
					},
				},
			},
		}
	}

	encodedMessages := func(t *testing.T, request *OpenAIChatRequest) string {
		t.Helper()
		body, err := json.Marshal(request.Messages)
		if err != nil {
			t.Fatalf("failed to marshal messages: %v", err)
		}
		return string(body)
	}

	t.Run("mistral carries no reasoning field on the wire", func(t *testing.T) {
		result := ToOpenAIChatRequest(ctx, buildRequest(schemas.Mistral))
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if len(result.Messages) != 2 || result.Messages[1].OpenAIChatAssistantMessage == nil {
			t.Fatalf("expected assistant message with OpenAI payload, got %#v", result.Messages)
		}
		assistant := result.Messages[1].OpenAIChatAssistantMessage
		if assistant.Reasoning != nil || assistant.ReasoningAlias != nil ||
			assistant.ReasoningText != nil || len(assistant.ReasoningDetails) > 0 {
			t.Fatalf("expected every reasoning spelling cleared, got %#v", assistant)
		}

		wire := encodedMessages(t, result)
		for _, key := range []string{"reasoning_content", `"reasoning"`, "reasoning_text", "reasoning_details"} {
			if strings.Contains(wire, key) {
				t.Fatalf("wire payload must not carry %s, got %s", key, wire)
			}
		}
	})

	t.Run("another provider still replays reasoning", func(t *testing.T) {
		result := ToOpenAIChatRequest(ctx, buildRequest(schemas.Fireworks))
		if result == nil || len(result.Messages) != 2 || result.Messages[1].OpenAIChatAssistantMessage == nil {
			t.Fatalf("expected assistant message with OpenAI payload, got %#v", result)
		}
		assistant := result.Messages[1].OpenAIChatAssistantMessage
		if assistant.Reasoning == nil || *assistant.Reasoning != reasoning {
			t.Fatalf("expected reasoning to survive for a non-Mistral provider, got %#v", assistant)
		}
	})
}
