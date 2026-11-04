package openai

import (
	"encoding/json"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// replayReasoningKeys are the four wire keys that carry assistant reasoning.
var replayReasoningKeys = []string{"reasoning_content", "reasoning", "reasoning_text", "reasoning_details"}

// reasoningWireOf marshals the first converted message and returns only the reasoning
// keys it carries, decoded. A key the wire does not carry is absent from the map.
func reasoningWireOf(t *testing.T, messages []OpenAIMessage) map[string]any {
	t.Helper()
	require.NotEmpty(t, messages)
	raw, err := sonic.Marshal(messages[0])
	require.NoError(t, err)
	var wire map[string]any
	require.NoError(t, json.Unmarshal(raw, &wire))
	got := map[string]any{}
	for _, key := range replayReasoningKeys {
		if v, ok := wire[key]; ok {
			got[key] = v
		}
	}
	return got
}

// replayAssistant converts one assistant message under a reasoning replay setting and
// returns the reasoning keys the wire carries.
func replayAssistant(t *testing.T, msg *schemas.ChatAssistantMessage, replayField string) map[string]any {
	t.Helper()
	return reasoningWireOf(t, convertBifrostMessagesToOpenAIMessages([]schemas.ChatMessage{{
		Role:                 schemas.ChatMessageRoleAssistant,
		ChatAssistantMessage: msg,
	}}, replayField))
}

// wantReasoning decodes an expected set of reasoning keys written as JSON.
func wantReasoning(t *testing.T, raw string) map[string]any {
	t.Helper()
	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &want))
	return want
}

const (
	replayText       = "TEXT-T"
	replayDetailText = "DETAIL-D"
	replaySignature  = "SIG-D"
	// replayDetailsJSON is replayDetails() as the wire carries it under reasoning_details.
	replayDetailsJSON = `[{"index":0,"type":"reasoning.text","text":"DETAIL-D","signature":"SIG-D"}]`
)

// replayDetails returns one reasoning_details entry with text and a signature.
func replayDetails() []schemas.ChatReasoningDetails {
	return []schemas.ChatReasoningDetails{{
		Index:     0,
		Type:      schemas.BifrostReasoningDetailsTypeText,
		Text:      schemas.Ptr(replayDetailText),
		Signature: schemas.Ptr(replaySignature),
	}}
}

// replayInputs are the four reasoning shapes an assistant message can arrive with.
var replayInputs = []struct {
	name string
	msg  func() *schemas.ChatAssistantMessage
}{
	{"text only", func() *schemas.ChatAssistantMessage {
		return &schemas.ChatAssistantMessage{Reasoning: schemas.Ptr(replayText), ReasoningSource: "reasoning_content"}
	}},
	{"details only", func() *schemas.ChatAssistantMessage {
		return &schemas.ChatAssistantMessage{ReasoningDetails: replayDetails()}
	}},
	{"both", func() *schemas.ChatAssistantMessage {
		return &schemas.ChatAssistantMessage{
			Reasoning:        schemas.Ptr(replayText),
			ReasoningSource:  "reasoning_content",
			ReasoningDetails: replayDetails(),
		}
	}},
	{"neither", func() *schemas.ChatAssistantMessage {
		return &schemas.ChatAssistantMessage{}
	}},
}

// TestConvertNonAssistantDetailsNeverReplayed checks the assistant-scoped rule on the wire:
// a user, system, or tool turn that carries only reasoning_details sends no reasoning key
// under any replay setting, reasoning_details included.
func TestConvertNonAssistantDetailsNeverReplayed(t *testing.T) {
	for _, role := range []schemas.ChatMessageRole{
		schemas.ChatMessageRoleUser,
		schemas.ChatMessageRoleSystem,
		schemas.ChatMessageRoleTool,
	} {
		for _, mode := range []string{
			schemas.ReasoningReplayReasoning,
			schemas.ReasoningReplayReasoningContent,
			schemas.ReasoningReplayReasoningText,
			schemas.ReasoningReplayReasoningDetails,
			schemas.ReasoningReplayNone,
			schemas.ReasoningReplayAuto,
		} {
			t.Run(string(role)+"/"+mode, func(t *testing.T) {
				got := reasoningWireOf(t, convertBifrostMessagesToOpenAIMessages([]schemas.ChatMessage{{
					Role:                 role,
					ChatAssistantMessage: &schemas.ChatAssistantMessage{ReasoningDetails: replayDetails()},
				}}, mode))
				assert.Empty(t, got, "a %s turn with only reasoning_details must send no reasoning key", role)
			})
		}
	}
}

// TestConvertAssistantReasoningReplayMatrix checks the exact reasoning keys the wire carries
// for each replay setting and each reasoning shape.
func TestConvertAssistantReasoningReplayMatrix(t *testing.T) {
	want := map[string]map[string]string{
		schemas.ReasoningReplayReasoning: {
			"text only":    `{"reasoning":"TEXT-T"}`,
			"details only": `{"reasoning":"DETAIL-D"}`,
			"both":         `{"reasoning":"TEXT-T"}`,
			"neither":      `{}`,
		},
		schemas.ReasoningReplayReasoningContent: {
			"text only":    `{"reasoning_content":"TEXT-T"}`,
			"details only": `{"reasoning_content":"DETAIL-D"}`,
			"both":         `{"reasoning_content":"TEXT-T"}`,
			"neither":      `{}`,
		},
		schemas.ReasoningReplayReasoningText: {
			"text only":    `{"reasoning_text":"TEXT-T"}`,
			"details only": `{"reasoning_text":"DETAIL-D"}`,
			"both":         `{"reasoning_text":"TEXT-T"}`,
			"neither":      `{}`,
		},
		// Details win when present. With no details, the message falls back to the auto
		// text rule, which picks reasoning_content for a reasoning_content source.
		schemas.ReasoningReplayReasoningDetails: {
			"text only":    `{"reasoning_content":"TEXT-T"}`,
			"details only": `{"reasoning_details":` + replayDetailsJSON + `}`,
			"both":         `{"reasoning_details":` + replayDetailsJSON + `}`,
			"neither":      `{}`,
		},
		schemas.ReasoningReplayNone: {
			"text only":    `{}`,
			"details only": `{}`,
			"both":         `{}`,
			"neither":      `{}`,
		},
		schemas.ReasoningReplayAuto: {
			"text only":    `{"reasoning_content":"TEXT-T"}`,
			"details only": `{"reasoning_content":"DETAIL-D"}`,
			"both":         `{"reasoning_content":"TEXT-T"}`,
			"neither":      `{}`,
		},
	}
	for _, mode := range []string{
		schemas.ReasoningReplayReasoning,
		schemas.ReasoningReplayReasoningContent,
		schemas.ReasoningReplayReasoningText,
		schemas.ReasoningReplayReasoningDetails,
		schemas.ReasoningReplayNone,
		schemas.ReasoningReplayAuto,
	} {
		for _, in := range replayInputs {
			t.Run(mode+"/"+in.name, func(t *testing.T) {
				got := replayAssistant(t, in.msg(), mode)
				assert.Equal(t, wantReasoning(t, want[mode][in.name]), got)
			})
		}
	}
}

// TestConvertAssistantReasoningDetailsPassThroughUnchanged checks that in the
// reasoning_details setting the caller's details reach the wire byte for byte, with the
// signature kept and no text field written, even when the message also carries Reasoning.
func TestConvertAssistantReasoningDetailsPassThroughUnchanged(t *testing.T) {
	id0, id1 := "rd_0", "rd_1"
	text0, sig0 := "step one", "sig-AAA"
	data1 := "enc-BBB"
	msg := &schemas.ChatAssistantMessage{
		Reasoning:       schemas.Ptr("TEXT-T"),
		ReasoningSource: "reasoning_content",
		ReasoningDetails: []schemas.ChatReasoningDetails{
			{ID: &id0, Index: 0, Type: schemas.BifrostReasoningDetailsTypeText, Text: &text0, Signature: &sig0},
			{ID: &id1, Index: 1, Type: schemas.BifrostReasoningDetailsTypeEncrypted, Data: &data1},
		},
	}
	out := convertBifrostMessagesToOpenAIMessages([]schemas.ChatMessage{{
		Role:                 schemas.ChatMessageRoleAssistant,
		ChatAssistantMessage: msg,
	}}, schemas.ReasoningReplayReasoningDetails)

	raw, err := sonic.Marshal(out[0])
	require.NoError(t, err)
	var wire struct {
		Details json.RawMessage `json:"reasoning_details"`
	}
	require.NoError(t, json.Unmarshal(raw, &wire))
	assert.Equal(t,
		`[{"id":"rd_0","index":0,"type":"reasoning.text","text":"step one","signature":"sig-AAA"},{"id":"rd_1","index":1,"type":"reasoning.encrypted","data":"enc-BBB"}]`,
		string(wire.Details))

	got := reasoningWireOf(t, out)
	assert.Len(t, got, 1, "only reasoning_details may be on the wire")
	assert.Contains(t, got, "reasoning_details")
}

// TestConvertAssistantReasoningAutoKeepsCallerSpelling checks that auto sends the text under
// the spelling the caller used, falls back to reasoning_content for an empty or
// reasoning_details source, and never sends reasoning_details even when details exist.
func TestConvertAssistantReasoningAutoKeepsCallerSpelling(t *testing.T) {
	cases := []struct {
		source string
		wire   string
	}{
		{"reasoning", "reasoning"},
		{"reasoning_content", "reasoning_content"},
		{"reasoning_text", "reasoning_text"},
		{"", "reasoning_content"},
		{"reasoning_details", "reasoning_content"},
	}
	for _, replayField := range []string{schemas.ReasoningReplayAuto, ""} {
		for _, tc := range cases {
			t.Run("replay="+replayField+" source="+tc.source, func(t *testing.T) {
				msg := &schemas.ChatAssistantMessage{
					Reasoning:        schemas.Ptr(replayText),
					ReasoningSource:  tc.source,
					ReasoningDetails: replayDetails(),
				}
				assert.Equal(t, map[string]any{tc.wire: replayText}, replayAssistant(t, msg, replayField))
			})
		}
	}
}

// TestConvertAssistantReasoningTextFoldsFirstNonEmptyDetail checks that, with no Reasoning,
// a text setting sends the first detail whose text is non-empty. An earlier detail with
// empty text and a signature must not win, and an empty Reasoning is treated as absent.
func TestConvertAssistantReasoningTextFoldsFirstNonEmptyDetail(t *testing.T) {
	empty, sig, second := "", "sig-EMPTY", "SECOND"
	details := []schemas.ChatReasoningDetails{
		{Index: 0, Type: schemas.BifrostReasoningDetailsTypeText, Text: &empty, Signature: &sig},
		{Index: 1, Type: schemas.BifrostReasoningDetailsTypeText, Text: &second},
	}
	cases := []struct {
		replayField string
		wire        string
	}{
		{schemas.ReasoningReplayReasoning, "reasoning"},
		{schemas.ReasoningReplayReasoningContent, "reasoning_content"},
		{schemas.ReasoningReplayReasoningText, "reasoning_text"},
		{schemas.ReasoningReplayAuto, "reasoning_content"},
	}
	for _, tc := range cases {
		t.Run(tc.replayField, func(t *testing.T) {
			msg := &schemas.ChatAssistantMessage{ReasoningDetails: details}
			assert.Equal(t, map[string]any{tc.wire: "SECOND"}, replayAssistant(t, msg, tc.replayField))
		})
	}

	t.Run("empty Reasoning is absent", func(t *testing.T) {
		for _, replayField := range []string{schemas.ReasoningReplayAuto, schemas.ReasoningReplayReasoningContent, schemas.ReasoningReplayReasoningDetails} {
			msg := &schemas.ChatAssistantMessage{Reasoning: schemas.Ptr("")}
			assert.Equal(t, map[string]any{}, replayAssistant(t, msg, replayField), "replay=%s", replayField)
		}
	})
}

// TestToOpenAIChatRequestReasoningReplayResolvesConfiguredField checks the resolved setting
// end to end through the request builder. The per-model map is keyed by the model the caller
// named: an aliased request is looked up by its alias, and a direct request by its model id.
func TestToOpenAIChatRequestReasoningReplayResolvesConfiguredField(t *testing.T) {
	assistant := func() []schemas.ChatMessage {
		return []schemas.ChatMessage{{
			Role: schemas.ChatMessageRoleAssistant,
			ChatAssistantMessage: &schemas.ChatAssistantMessage{
				Reasoning:        schemas.Ptr(replayText),
				ReasoningSource:  "reasoning_content",
				ReasoningDetails: replayDetails(),
			},
		}}
	}
	cases := []struct {
		name   string
		model  string
		alias  string
		custom *schemas.CustomProviderConfig
		want   string
	}{
		{
			name:   "model id matches a reasoning_text entry",
			model:  "glm",
			custom: &schemas.CustomProviderConfig{ReasoningReplayFields: map[string]string{"glm": schemas.ReasoningReplayReasoningText}},
			want:   `{"reasoning_text":"TEXT-T"}`,
		},
		{
			name:   "model id matches a reasoning_details entry",
			model:  "kimi",
			custom: &schemas.CustomProviderConfig{ReasoningReplayFields: map[string]string{"kimi": schemas.ReasoningReplayReasoningDetails}},
			want:   `{"reasoning_details":` + replayDetailsJSON + `}`,
		},
		{
			name:   "model with no entry falls back to auto",
			model:  "other",
			custom: &schemas.CustomProviderConfig{ReasoningReplayFields: map[string]string{"glm": schemas.ReasoningReplayReasoningText}},
			want:   `{"reasoning_content":"TEXT-T"}`,
		},
		{
			// Bifrost rewrites the model to the upstream id once the alias resolves, so the
			// entry must be found by the caller's alias "glm", not by "glm-5.3-flash".
			name:   "aliased request is looked up by the caller's alias",
			model:  "glm-5.3-flash",
			alias:  "glm",
			custom: &schemas.CustomProviderConfig{ReasoningReplayFields: map[string]string{"glm": schemas.ReasoningReplayReasoningText}},
			want:   `{"reasoning_text":"TEXT-T"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
			if tc.alias != "" {
				ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{Key: tc.alias})
			}
			req := toOpenAIChatRequest(ctx, &schemas.BifrostChatRequest{
				Provider: schemas.OpenAI,
				Model:    tc.model,
				Input:    assistant(),
			}, tc.custom)
			require.NotNil(t, req)
			assert.Equal(t, wantReasoning(t, tc.want), reasoningWireOf(t, req.Messages))
		})
	}
}
