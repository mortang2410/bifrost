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

	t.Run("a caller-supplied thinking chunk absorbs the sibling reasoning", func(t *testing.T) {
		name := "get_weather"
		callID := "call_1"
		callType := "function"
		clientThought := "client-side thought"
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
					Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
						{Type: schemas.ChatContentBlockTypeThinking, Thinking: []schemas.ChatContentBlock{
							{Type: schemas.ChatContentBlockTypeText, Text: &clientThought},
						}},
						{Type: schemas.ChatContentBlockTypeText, Text: &answer},
					}},
					ChatAssistantMessage: &schemas.ChatAssistantMessage{
						Reasoning: &reasoning,
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
		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks) != 2 {
			t.Fatalf("expected the caller's blocks to survive, got %#v", content)
		}
		if content.ContentBlocks[0].Type != schemas.ChatContentBlockTypeThinking {
			t.Fatalf("expected one leading thinking block, got %#v", content.ContentBlocks[0])
		}
		thoughts := content.ContentBlocks[0].Thinking
		if len(thoughts) != 2 || thoughts[0].Text == nil || *thoughts[0].Text != clientThought ||
			thoughts[1].Text == nil || *thoughts[1].Text != reasoning {
			t.Fatalf("expected the sibling reasoning merged into the existing chunk, got %#v", thoughts)
		}
		if len(result.Messages[1].OpenAIChatAssistantMessage.ToolCalls) != 1 {
			t.Fatalf("expected the tool call to survive replay, got %#v", result.Messages[1].OpenAIChatAssistantMessage.ToolCalls)
		}
		wire := encodedMessages(t, result)
		if !strings.Contains(wire, `"thinking":[{"type":"text","text":"client-side thought"},{"type":"text","text":"seven times six"}]`) {
			t.Fatalf("expected both texts in one leading thinking chunk, got %s", wire)
		}
		for _, key := range []string{"reasoning_content", "reasoning_text", "reasoning_details"} {
			if strings.Contains(wire, key) {
				t.Fatalf("wire payload must not carry %s, got %s", key, wire)
			}
		}
	})

	t.Run("two caller-supplied thinking chunks fold into one", func(t *testing.T) {
		first := "first caller thought"
		second := "second caller thought"
		request := &schemas.BifrostChatRequest{
			Provider: schemas.Mistral,
			Model:    "deepseek-v4-1-flash-1017",
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &userContent}},
				{
					Role: schemas.ChatMessageRoleAssistant,
					Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
						{Type: schemas.ChatContentBlockTypeThinking, Thinking: []schemas.ChatContentBlock{
							{Type: schemas.ChatContentBlockTypeText, Text: &first},
						}},
						{Type: schemas.ChatContentBlockTypeThinking, Thinking: []schemas.ChatContentBlock{
							{Type: schemas.ChatContentBlockTypeText, Text: &second},
						}},
						{Type: schemas.ChatContentBlockTypeText, Text: &answer},
					}},
					ChatAssistantMessage: &schemas.ChatAssistantMessage{Reasoning: &reasoning},
				},
			},
		}

		result := ToOpenAIChatRequest(ctx, request)
		if result == nil || len(result.Messages) != 2 {
			t.Fatalf("expected two messages, got %#v", result)
		}
		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks) != 2 ||
			content.ContentBlocks[0].Type != schemas.ChatContentBlockTypeThinking {
			t.Fatalf("expected one thinking block plus the answer, got %#v", content)
		}
		thoughts := content.ContentBlocks[0].Thinking
		if len(thoughts) != 3 || *thoughts[0].Text != first || *thoughts[1].Text != second || *thoughts[2].Text != reasoning {
			t.Fatalf("expected all three texts under one leading chunk, got %#v", thoughts)
		}
	})

	t.Run("identical reasoning is not replayed twice", func(t *testing.T) {
		request := &schemas.BifrostChatRequest{
			Provider: schemas.Mistral,
			Model:    "deepseek-v4-1-flash-1017",
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &userContent}},
				{
					Role: schemas.ChatMessageRoleAssistant,
					Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
						{Type: schemas.ChatContentBlockTypeThinking, Thinking: []schemas.ChatContentBlock{
							{Type: schemas.ChatContentBlockTypeText, Text: &reasoning},
						}},
						{Type: schemas.ChatContentBlockTypeText, Text: &answer},
					}},
					ChatAssistantMessage: &schemas.ChatAssistantMessage{Reasoning: &reasoning},
				},
			},
		}

		result := ToOpenAIChatRequest(ctx, request)
		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks[0].Thinking) != 1 {
			t.Fatalf("expected the duplicate text to be skipped, got %#v", content)
		}
	})

	t.Run("every non-empty reasoning detail is replayed", func(t *testing.T) {
		firstDetail := "first detail"
		secondDetail := "second detail"
		emptyDetail := ""
		cases := []struct {
			label    string
			details  []*string
			expected []string
		}{
			{"two texts", []*string{&firstDetail, &secondDetail}, []string{firstDetail, secondDetail}},
			{"empty first text", []*string{&emptyDetail, &secondDetail}, []string{secondDetail}},
		}
		// The details spelling is what a provider config names when upstream accepts
		// reasoning_details; the fold must not keep only the first text.
		custom := &schemas.CustomProviderConfig{ReasoningReplayField: schemas.ReasoningReplayReasoningDetails}
		for _, tc := range cases {
			reasoningDetails := make([]schemas.ChatReasoningDetails, 0, len(tc.details))
			for index, text := range tc.details {
				reasoningDetails = append(reasoningDetails, schemas.ChatReasoningDetails{Index: index, Type: schemas.BifrostReasoningDetailsTypeText, Text: text})
			}
			request := &schemas.BifrostChatRequest{
				Provider: schemas.Mistral,
				Model:    "deepseek-v4-1-flash-1017",
				Input: []schemas.ChatMessage{
					{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &userContent}},
					{
						Role:                 schemas.ChatMessageRoleAssistant,
						Content:              &schemas.ChatMessageContent{ContentStr: &answer},
						ChatAssistantMessage: &schemas.ChatAssistantMessage{ReasoningDetails: reasoningDetails},
					},
				},
			}

			result := ToOpenAIChatRequestForProvider(ctx, request, custom)
			content := result.Messages[1].Content
			if content == nil || len(content.ContentBlocks) == 0 ||
				content.ContentBlocks[0].Type != schemas.ChatContentBlockTypeThinking {
				t.Fatalf("%s: expected a thinking block, got %#v", tc.label, content)
			}
			thoughts := content.ContentBlocks[0].Thinking
			if len(thoughts) != len(tc.expected) {
				t.Fatalf("%s: expected %d texts, got %#v", tc.label, len(tc.expected), thoughts)
			}
			for index, text := range tc.expected {
				if thoughts[index].Text == nil || *thoughts[index].Text != text {
					t.Fatalf("%s: expected %q at %d, got %#v", tc.label, text, index, thoughts)
				}
			}
		}
	})

	t.Run("an empty caller thinking chunk still receives the sibling", func(t *testing.T) {
		request := &schemas.BifrostChatRequest{
			Provider: schemas.Mistral,
			Model:    "deepseek-v4-1-flash-1017",
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &userContent}},
				{
					Role: schemas.ChatMessageRoleAssistant,
					Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
						{Type: schemas.ChatContentBlockTypeThinking},
						{Type: schemas.ChatContentBlockTypeText, Text: &answer},
					}},
					ChatAssistantMessage: &schemas.ChatAssistantMessage{Reasoning: &reasoning},
				},
			},
		}

		result := ToOpenAIChatRequest(ctx, request)
		content := result.Messages[1].Content
		if content == nil || len(content.ContentBlocks) != 2 {
			t.Fatalf("expected the caller's blocks, got %#v", content)
		}
		thoughts := content.ContentBlocks[0].Thinking
		if len(thoughts) != 1 || thoughts[0].Text == nil || *thoughts[0].Text != reasoning {
			t.Fatalf("expected the sibling text in the empty chunk, got %#v", thoughts)
		}
	})
}

// Mistral documents a closed reasoning_effort enum (none, minimal, low, medium,
// high, xhigh, max) and answers an out-of-enum value with a validation error. The
// gateway must forward the caller's level rather than collapsing it to one it
// considers supported.
func TestToOpenAIChatRequest_MistralForwardsDocumentedReasoningEffort(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(nil)
	defer cancel()

	userContent := "What is 7*6?"
	for _, effort := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		request := &schemas.BifrostChatRequest{
			Provider: schemas.Mistral,
			Model:    "deepseek-v4-1-flash-1017",
			Params:   &schemas.ChatParameters{Reasoning: &schemas.ChatReasoning{Effort: &effort}},
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &userContent}},
			},
		}

		result := ToOpenAIChatRequest(ctx, request)
		if result == nil || result.Reasoning == nil || result.Reasoning.Effort == nil {
			t.Fatalf("%s: expected the effort to survive, got %#v", effort, result)
		}
		if got := *result.Reasoning.Effort; got != effort {
			t.Fatalf("%s: expected the effort forwarded unchanged, got %q", effort, got)
		}
	}
}
