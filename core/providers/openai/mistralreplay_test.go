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
// carry its reasoning inside the content array and no reasoning_content,
// reasoning, reasoning_text or reasoning_details beside it.
func TestToOpenAIChatRequest_MistralReplaysReasoningAsThinkingChunk(t *testing.T) {
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

	t.Run("mistral replays reasoning as a thinking content chunk", func(t *testing.T) {
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

		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks) != 2 {
			t.Fatalf("expected a thinking block and the answer block, got %#v", content)
		}
		thinking := content.ContentBlocks[0]
		if thinking.Type != schemas.ChatContentBlockTypeThinking || len(thinking.Thinking) != 1 ||
			thinking.Thinking[0].Text == nil || *thinking.Thinking[0].Text != reasoning {
			t.Fatalf("expected the reasoning inside a thinking block, got %#v", thinking)
		}
		answerBlock := content.ContentBlocks[1]
		if answerBlock.Type != schemas.ChatContentBlockTypeText || answerBlock.Text == nil || *answerBlock.Text != answer {
			t.Fatalf("expected the answer text after the thinking block, got %#v", answerBlock)
		}

		wire := encodedMessages(t, result)
		for _, key := range []string{"reasoning_content", "reasoning_text", "reasoning_details"} {
			if strings.Contains(wire, key) {
				t.Fatalf("wire payload must not carry %s, got %s", key, wire)
			}
		}
		if !strings.Contains(wire, `"thinking":[{"type":"text","text":"seven times six"}]`) {
			t.Fatalf("expected the thinking chunk on the wire, got %s", wire)
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

	t.Run("tool-call turn keeps its calls beside the thinking block", func(t *testing.T) {
		name := "get_weather"
		callID := "call_1"
		callType := "function"
		toolReasoning := "I should check the weather"
		request := &schemas.BifrostChatRequest{
			Provider: schemas.Mistral,
			Model:    "deepseek-v4-1-flash-1017",
			Input: []schemas.ChatMessage{
				{
					Role:    schemas.ChatMessageRoleUser,
					Content: &schemas.ChatMessageContent{ContentStr: &userContent},
				},
				{
					Role: schemas.ChatMessageRoleAssistant,
					ChatAssistantMessage: &schemas.ChatAssistantMessage{
						Reasoning: &toolReasoning,
						ToolCalls: []schemas.ChatAssistantMessageToolCall{{
							Type:     &callType,
							ID:       &callID,
							Function: schemas.ChatAssistantMessageToolCallFunction{Name: &name, Arguments: "{}"},
						}},
					},
				},
			},
		}

		result := ToOpenAIChatRequest(ctx, request)
		if result == nil || len(result.Messages) != 2 || result.Messages[1].OpenAIChatAssistantMessage == nil {
			t.Fatalf("expected assistant message with OpenAI payload, got %#v", result)
		}
		assistant := result.Messages[1].OpenAIChatAssistantMessage
		if len(assistant.ToolCalls) != 1 {
			t.Fatalf("expected the tool call to survive replay, got %#v", assistant.ToolCalls)
		}
		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks) != 1 ||
			content.ContentBlocks[0].Type != schemas.ChatContentBlockTypeThinking {
			t.Fatalf("expected only a thinking block when the turn carried no text, got %#v", content)
		}
	})
}
