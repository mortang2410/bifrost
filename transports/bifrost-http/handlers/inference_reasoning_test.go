package handlers

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/openai"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// nativeChatRequest decodes a body the way the native POST /v1/chat/completions
// route does. prepareRequest[ChatRequest] calls sonic.Unmarshal on the request
// body, which invokes ChatRequest.UnmarshalJSON, so decoding into ChatRequest
// here exercises the real ingress seam rather than a stand-in for it.
func nativeChatRequest(t *testing.T, body string) *ChatRequest {
	t.Helper()
	var req ChatRequest
	require.NoError(t, sonic.Unmarshal([]byte(body), &req))
	return &req
}

// TestNativeChatRequestPromotesReasoningDetailsText covers the native ingress
// path, which decodes messages straight into the Bifrost schema instead of going
// through the OpenAI integration converter. A caller that replays its assistant
// reasoning only as reasoning_details must still reach provider request
// conversion with Reasoning set, because DeepSeek-shaped upstreams reject an
// assistant tool-call turn that carries no reasoning_content.
func TestNativeChatRequestPromotesReasoningDetailsText(t *testing.T) {
	const reasoning = "RD-MARK"
	req := nativeChatRequest(t, `{"model":"deepseek-v4.1-flash","messages":[{"role":"assistant","reasoning_details":[{"index":0,"type":"reasoning.text","text":"RD-MARK","summary":"summary marker","signature":"sig marker","data":"data marker"}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}`)

	require.Len(t, req.Messages, 1)
	assistant := req.Messages[0].ChatAssistantMessage
	require.NotNil(t, assistant)
	require.NotNil(t, assistant.Reasoning,
		"native ingress must normalize reasoning_details text for provider replay")
	assert.Equal(t, reasoning, *assistant.Reasoning)
	require.Len(t, assistant.ReasoningDetails, 1,
		"promoting text must preserve the original reasoning detail")
	require.NotNil(t, assistant.ReasoningDetails[0].Text)
	require.NotNil(t, assistant.ReasoningDetails[0].Summary)
	require.NotNil(t, assistant.ReasoningDetails[0].Signature)
	require.NotNil(t, assistant.ReasoningDetails[0].Data)
	assert.Equal(t, reasoning, *assistant.ReasoningDetails[0].Text)
	assert.Equal(t, "summary marker", *assistant.ReasoningDetails[0].Summary)
	assert.Equal(t, "sig marker", *assistant.ReasoningDetails[0].Signature)
	assert.Equal(t, "data marker", *assistant.ReasoningDetails[0].Data)
	require.Len(t, assistant.ToolCalls, 1, "tool-call history must survive native ingress")
}

// TestNativeChatRequestOnlyPromotesAssistantReasoning pins the role gate.
// Reasoning is an assistant-only concept, so a user, system, or tool turn must
// never gain it; otherwise the message reaches the provider with a
// reasoning_content key its schema has no field for.
func TestNativeChatRequestOnlyPromotesAssistantReasoning(t *testing.T) {
	cases := map[string]string{
		"user":   `{"model":"m","messages":[{"role":"user","reasoning_details":[{"index":0,"type":"reasoning.text","text":"USER-MARK"}]}]}`,
		"system": `{"model":"m","messages":[{"role":"system","reasoning_details":[{"index":0,"type":"reasoning.text","text":"SYS-MARK"}]}]}`,
		"tool":   `{"model":"m","messages":[{"role":"tool","tool_call_id":"c1","reasoning_details":[{"index":0,"type":"reasoning.text","text":"TOOL-MARK"}]}]}`,
	}
	for role, body := range cases {
		t.Run(role, func(t *testing.T) {
			req := nativeChatRequest(t, body)
			require.Len(t, req.Messages, 1)
			assistant := req.Messages[0].ChatAssistantMessage
			require.NotNil(t, assistant, "the reasoning_details entry must still be parsed")
			assert.Nil(t, assistant.Reasoning,
				"a %s turn must not have reasoning_details text promoted onto Reasoning", role)

			ctx, cancel := schemas.NewBifrostContextWithCancel(nil)
			defer cancel()
			wireBody, err := sonic.Marshal(openai.ToOpenAIChatRequest(ctx, &schemas.BifrostChatRequest{
				Provider: schemas.OpenAI,
				Model:    "m",
				Input:    req.Messages,
			}))
			require.NoError(t, err)
			var decoded map[string]any
			require.NoError(t, sonic.Unmarshal(wireBody, &decoded))
			messages := decoded["messages"].([]any)
			require.Len(t, messages, 1)
			assert.NotContains(t, messages[0], "reasoning_content",
				"a %s turn must not reach the provider with reasoning_content", role)
			assert.NotContains(t, messages[0], "reasoning",
				"a %s turn must not reach the provider with reasoning", role)
		})
	}
}

// TestNativeChatRequestKeepsExplicitReasoningForAnyRole pins the exact scope of the
// role gate: only the reasoning_details promotion is assistant-scoped. An explicit
// reasoning or reasoning_content value survives on any role, because the shared
// schema decoder folds it whatever the role, and the integration mount matches
// that. Widening the gate to the explicit value would drop replayed reasoning.
func TestNativeChatRequestKeepsExplicitReasoningForAnyRole(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"user":      {body: `{"model":"m","messages":[{"role":"user","reasoning_content":"RC-MARK"}]}`, want: "RC-MARK"},
		"system":    {body: `{"model":"m","messages":[{"role":"system","reasoning":"R-MARK"}]}`, want: "R-MARK"},
		"tool":      {body: `{"model":"m","messages":[{"role":"tool","tool_call_id":"c1","reasoning_content":"RC-MARK"}]}`, want: "RC-MARK"},
		"no role":   {body: `{"model":"m","messages":[{"reasoning_content":"RC-MARK"}]}`, want: "RC-MARK"},
		"assistant": {body: `{"model":"m","messages":[{"role":"assistant","reasoning_content":"RC-MARK"}]}`, want: "RC-MARK"},
	}
	for role, tc := range cases {
		t.Run(role, func(t *testing.T) {
			req := nativeChatRequest(t, tc.body)
			require.Len(t, req.Messages, 1)
			assistant := req.Messages[0].ChatAssistantMessage
			require.NotNil(t, assistant)
			require.NotNil(t, assistant.Reasoning,
				"an explicit reasoning value must survive native ingress regardless of role")
			// Assert the exact value, not merely that something was selected: a fold
			// that returned the wrong text would otherwise pass.
			assert.Equal(t, tc.want, *assistant.Reasoning,
				"native ingress must keep the caller's explicit reasoning value verbatim")
		})
	}
}

// reasoningDetailExpect describes one reasoning_details entry exactly as the
// caller sent it; a nil pointer means that JSON key was absent. Comparing whole
// entries rather than a count is what catches a normalization pass that rewrites
// or drops the caller's metadata, which a count assertion cannot see.
type reasoningDetailExpect struct {
	index     int
	typ       schemas.BifrostReasoningDetailsType
	text      *string
	summary   *string
	signature *string
	data      *string
}

// textDetailExpect is the common single-text detail.
func textDetailExpect(index int, text string) reasoningDetailExpect {
	return reasoningDetailExpect{index: index, typ: schemas.BifrostReasoningDetailsTypeText, text: &text}
}

func expectStr(s string) *string { return &s }

// assertReasoningDetails compares the decoded details against the caller's
// entries field by field, so a pass that keeps the count but rewrites text,
// summary, signature, or data fails.
func assertReasoningDetails(t *testing.T, want []reasoningDetailExpect, got []schemas.ChatReasoningDetails, context string) {
	t.Helper()
	require.Len(t, got, len(want), "%s: reasoning_details must be preserved exactly", context)
	for i := range want {
		w, g := want[i], got[i]
		assert.Equal(t, w.index, g.Index, "%s: detail %d index", context, i)
		assert.Equal(t, w.typ, g.Type, "%s: detail %d type", context, i)
		assert.Equal(t, w.text, g.Text, "%s: detail %d text", context, i)
		assert.Equal(t, w.summary, g.Summary, "%s: detail %d summary", context, i)
		assert.Equal(t, w.signature, g.Signature, "%s: detail %d signature", context, i)
		assert.Equal(t, w.data, g.Data, "%s: detail %d data", context, i)
	}
}

// reasoningSelectionCase is one row of the shared case table: a name and the raw
// caller message. What the row must produce is computed from the input by
// referenceSelect, never stored.
type reasoningSelectionCase struct {
	name string
	raw  string
}

// reasoningSelectionFixturePath is the one canonical case table both ingress
// paths are tested against, relative to this package directory.
//
// The fixture stores inputs only. What each input must produce is computed by
// the reference rule below, never stored beside the input, so editing the
// fixture cannot bless wrong behavior: it changes the input, and the reference
// recomputes the expectation from that input. A name-to-shape check, also code
// rather than data, fails if a row's input stops exercising the scenario its
// name promises.
const reasoningSelectionFixturePath = "../../../testdata/reasoning_selection_cases.json"

// reasoningSelectionFixture is one row of the shared table: a name and the raw
// caller message.
type reasoningSelectionFixture struct {
	Name string `json:"name"`
	Raw  string `json:"raw"`
}

// reasoningSelectionCaseNames is the canonical set of fixture rows, in order.
var reasoningSelectionCaseNames = []string{
	"details only",
	"empty first detail then text",
	"empty reasoning_content then detail text",
	"empty reasoning then detail text",
	"explicit reasoning_content wins",
	"explicit reasoning wins",
	"reasoning_content wins over reasoning",
	"non-empty reasoning_content with empty reasoning and details",
	"empty reasoning with non-empty reasoning_content and no details",
	"empty reasoning with non-empty reasoning_content on a user turn",
	"signature-only detail with no text",
	"both explicit empty",
	"explicit reasoning_content with empty details array",
	"both explicit non-empty with details",
}

// reasoningSelectionSelectsNothing names the rows whose input must select no
// reasoning. It is checked against the reference rule's own verdict, so a
// fixture edit that makes such a row start selecting fails.
var reasoningSelectionSelectsNothing = []string{
	"signature-only detail with no text",
	"both explicit empty",
}

// parsedFixtureInput is the caller message decoded for the reference rule.
type parsedFixtureInput struct {
	Role             *string         `json:"role"`
	Reasoning        *string         `json:"reasoning"`
	ReasoningContent *string         `json:"reasoning_content"`
	ReasoningDetails json.RawMessage `json:"reasoning_details"`
}

// fixtureDetail is one reasoning_details entry as the caller sent it.
type fixtureDetail struct {
	Index     int     `json:"index"`
	Type      string  `json:"type"`
	Text      *string `json:"text"`
	Summary   *string `json:"summary"`
	Signature *string `json:"signature"`
	Data      *string `json:"data"`
}

// referenceSelection is what the documented rule says one input must produce,
// derived from the input alone.
type referenceSelection struct {
	// selection is the reasoning text that must be selected, or nil for none.
	selection *string
	// callerDetails are the details the caller sent, unchanged; nil when the
	// caller sent no non-empty reasoning_details array.
	callerDetails []reasoningDetailExpect
	// sentDetailsArray records whether the caller supplied a non-empty details
	// array. An explicitly empty array counts as absent: it supplies no detail
	// text, so the native seam synthesizes one detail from the winning value.
	sentDetailsArray bool
}

// referenceFirstNonEmpty returns the first present, non-empty candidate, or nil.
// An empty string counts as absent.
func referenceFirstNonEmpty(candidates ...*string) *string {
	for _, c := range candidates {
		if c != nil && *c != "" {
			return c
		}
	}
	return nil
}

// referenceSelect implements the documented selection rule independently of the
// production code: first non-empty of reasoning_content then reasoning, then, on
// an assistant turn only, the first non-empty detail text. An empty explicit
// value counts as absent. The caller's details are returned unchanged.
func referenceSelect(t *testing.T, raw string) referenceSelection {
	t.Helper()
	var in parsedFixtureInput
	require.NoError(t, sonic.Unmarshal([]byte(raw), &in), "fixture input must be valid JSON")

	// An explicitly empty array is treated as no array: it supplies no detail
	// text, so the native seam synthesizes one detail from the winning value.
	var decoded []fixtureDetail
	if in.ReasoningDetails != nil {
		require.NoError(t, sonic.Unmarshal(in.ReasoningDetails, &decoded),
			"fixture reasoning_details must be an array")
	}
	ref := referenceSelection{sentDetailsArray: len(decoded) > 0}
	if ref.sentDetailsArray {
		ref.callerDetails = make([]reasoningDetailExpect, 0, len(decoded))
		for _, d := range decoded {
			ref.callerDetails = append(ref.callerDetails, reasoningDetailExpect{
				index:     d.Index,
				typ:       schemas.BifrostReasoningDetailsType(d.Type),
				text:      d.Text,
				summary:   d.Summary,
				signature: d.Signature,
				data:      d.Data,
			})
		}
	}

	ref.selection = referenceFirstNonEmpty(in.ReasoningContent, in.Reasoning)
	if ref.selection == nil && in.Role != nil && *in.Role == string(schemas.ChatMessageRoleAssistant) {
		for _, d := range ref.callerDetails {
			if d.text != nil && *d.text != "" {
				text := *d.text
				ref.selection = &text
				break
			}
		}
	}
	return ref
}

// referenceInputShape asserts that a row's input still exercises the scenario its
// name promises. This is code, not fixture data, so a fixture edit that empties a
// scenario fails here instead of passing quietly.
func referenceInputShape(t *testing.T, name string, in parsedFixtureInput, details []reasoningDetailExpect, sentArray bool) {
	t.Helper()

	emptyPresent := func(p *string) bool { return p != nil && *p == "" }
	nonEmpty := func(p *string) bool { return p != nil && *p != "" }
	firstTextEmpty := func() bool {
		return len(details) >= 1 && (details[0].text == nil || *details[0].text == "")
	}
	firstTextNonEmpty := func() bool {
		return len(details) >= 1 && details[0].text != nil && *details[0].text != ""
	}
	isAssistant := in.Role != nil && *in.Role == string(schemas.ChatMessageRoleAssistant)

	fail := func(why string) {
		t.Fatalf("row %q no longer exercises its scenario: %s", name, why)
	}

	switch name {
	case "details only":
		if in.Reasoning != nil || in.ReasoningContent != nil || !firstTextNonEmpty() {
			fail("expected no explicit spelling and a first detail with non-empty text")
		}
	case "empty first detail then text":
		if in.Reasoning != nil || in.ReasoningContent != nil || len(details) < 2 ||
			!firstTextEmpty() || details[0].signature == nil || !firstTextNonEmptyAt(details, 1) {
			fail("expected an empty first detail with a signature followed by a non-empty detail")
		}
	case "empty reasoning_content then detail text":
		if !emptyPresent(in.ReasoningContent) || !firstTextNonEmpty() {
			fail("expected an empty reasoning_content and a non-empty detail text")
		}
	case "empty reasoning then detail text":
		if !emptyPresent(in.Reasoning) || !firstTextNonEmpty() {
			fail("expected an empty reasoning and a non-empty detail text")
		}
	case "explicit reasoning_content wins":
		// The explicit value and the detail text compete here, so they must
		// differ: if they were equal, reversing the precedence would still pass.
		if !nonEmpty(in.ReasoningContent) || len(details) == 0 ||
			!firstTextNonEmpty() || *in.ReasoningContent == *details[0].text {
			fail("expected a non-empty reasoning_content that differs from the first detail text")
		}
	case "explicit reasoning wins":
		if !nonEmpty(in.Reasoning) || in.ReasoningContent != nil || len(details) == 0 ||
			!firstTextNonEmpty() || *in.Reasoning == *details[0].text {
			fail("expected a non-empty reasoning that differs from the first detail text")
		}
	case "reasoning_content wins over reasoning":
		// Both spellings compete, so they must differ: if they were equal,
		// reversing the precedence would still pass.
		if !nonEmpty(in.ReasoningContent) || !nonEmpty(in.Reasoning) || sentArray ||
			*in.ReasoningContent == *in.Reasoning {
			fail("expected both explicit spellings non-empty, differing, and no details array")
		}
	case "non-empty reasoning_content with empty reasoning and details":
		if !nonEmpty(in.ReasoningContent) || !emptyPresent(in.Reasoning) || len(details) == 0 {
			fail("expected non-empty reasoning_content with an empty reasoning and details")
		}
	case "empty reasoning with non-empty reasoning_content and no details":
		if !isAssistant || !emptyPresent(in.Reasoning) || !nonEmpty(in.ReasoningContent) || sentArray {
			fail("expected an assistant turn with an empty reasoning, non-empty reasoning_content, no details")
		}
	case "empty reasoning with non-empty reasoning_content on a user turn":
		if in.Role == nil || *in.Role != string(schemas.ChatMessageRoleUser) ||
			!emptyPresent(in.Reasoning) || !nonEmpty(in.ReasoningContent) {
			fail("expected a user turn with an empty reasoning and a non-empty reasoning_content")
		}
	case "signature-only detail with no text":
		if in.Reasoning != nil || in.ReasoningContent != nil || len(details) != 1 ||
			!firstTextEmpty() || details[0].signature == nil || details[0].summary != nil || details[0].data != nil {
			fail("expected one signature-only detail with no text, summary, or data")
		}
	case "both explicit empty":
		if !emptyPresent(in.Reasoning) || !emptyPresent(in.ReasoningContent) || sentArray {
			fail("expected both explicit spellings present and empty, with no details array")
		}
	case "explicit reasoning_content with empty details array":
		// The array must be present and empty, so this row keeps pinning that an
		// explicitly empty array is treated as no array.
		if !nonEmpty(in.ReasoningContent) || in.ReasoningDetails == nil ||
			sentArray || len(details) != 0 {
			fail("expected a non-empty reasoning_content with an explicitly empty details array")
		}
	case "both explicit non-empty with details":
		// Both spellings compete, so they must differ, and the winner must also
		// differ from the detail text, so every precedence edge stays detectable.
		if !nonEmpty(in.Reasoning) || !nonEmpty(in.ReasoningContent) || len(details) == 0 ||
			!firstTextNonEmpty() || *in.ReasoningContent == *in.Reasoning ||
			*in.ReasoningContent == *details[0].text {
			fail("expected both explicit spellings non-empty and differing, with differing details")
		}
	default:
		t.Fatalf("row %q has no shape check; add one so its scenario cannot be emptied", name)
	}
}

// firstTextNonEmptyAt reports whether the detail at i carries non-empty text.
func firstTextNonEmptyAt(details []reasoningDetailExpect, i int) bool {
	return i < len(details) && details[i].text != nil && *details[i].text != ""
}

// normalizeFixtureInput renders a fixture input in a canonical form, so two
// inputs that differ only in whitespace or key order compare equal.
//
// Numbers are kept as their original literals via UseNumber. Decoding them into
// any would convert them to float64, which rounds distinct large integers to the
// same value.
func normalizeFixtureInput(t *testing.T, raw string) string {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	var decoded any
	require.NoError(t, dec.Decode(&decoded), "fixture input must be valid JSON")
	out, err := json.Marshal(decoded)
	require.NoError(t, err)
	return string(out)
}

// loadReasoningSelectionCases reads the shared fixture and validates it: the row
// names must be exactly the canonical set in order, every input must be distinct
// after normalization, and every input must still exercise the scenario its name
// promises. The expectations themselves are not read from the file; the tests
// compute them with referenceSelect.
func loadReasoningSelectionCases(t *testing.T) []reasoningSelectionCase {
	t.Helper()
	data, err := os.ReadFile(reasoningSelectionFixturePath)
	require.NoError(t, err, "the shared reasoning selection fixture must be readable")
	var rows []reasoningSelectionFixture
	require.NoError(t, sonic.Unmarshal(data, &rows), "the shared fixture must be valid JSON")

	names := make([]string, 0, len(rows))
	inputs := map[string]string{}
	cases := make([]reasoningSelectionCase, 0, len(rows))
	for _, r := range rows {
		names = append(names, r.Name)
		require.NotEmpty(t, r.Raw, "row %q must carry an input", r.Name)

		normalized := normalizeFixtureInput(t, r.Raw)
		if prior, dup := inputs[normalized]; dup {
			t.Fatalf("rows %q and %q carry the same input; a duplicate row drops coverage", prior, r.Name)
		}
		inputs[normalized] = r.Name

		var in parsedFixtureInput
		require.NoError(t, sonic.Unmarshal([]byte(r.Raw), &in), "row %q input must be valid JSON", r.Name)
		ref := referenceSelect(t, r.Raw)
		referenceInputShape(t, r.Name, in, ref.callerDetails, ref.sentDetailsArray)

		cases = append(cases, reasoningSelectionCase{name: r.Name, raw: r.Raw})
	}

	require.Equal(t, reasoningSelectionCaseNames, names,
		"the shared fixture must carry exactly the canonical rows, in order")
	return cases
}

// assertReferenceSelectsNothing checks the reference rule's verdict for the rows
// that must select no reasoning, so a fixture edit that makes one of them start
// selecting fails even though the reference recomputes from the new input.
func assertReferenceSelectsNothing(t *testing.T) {
	t.Helper()
	cases := loadReasoningSelectionCases(t)
	got := map[string]bool{}
	for _, tc := range cases {
		if referenceSelect(t, tc.raw).selection == nil {
			got[tc.name] = true
		}
	}
	for _, name := range reasoningSelectionSelectsNothing {
		assert.True(t, got[name], "row %q must select no reasoning", name)
		delete(got, name)
	}
	assert.Empty(t, got, "these rows select no reasoning but are not listed as such: %v", got)
}

// TestNativeChatIngressReasoningSelection is the native-ingress half of a pair of
// tests that pin one selection rule: the first non-empty reasoning text wins, and
// an empty explicit value counts as absent. The integration-mount half lives in
// core/providers/openai (TestOpenAIConverterReasoningSelection) because the
// transports module resolves its core dependency from the module cache in a plain
// module build. Both read the same fixture.
//
// The expectation is computed by referenceSelect from the input alone, so it is
// the documented rule rather than a stored value, and each row asserts the whole
// decoded detail list so a pass that rewrites or drops caller metadata fails.
func TestNativeChatIngressReasoningSelection(t *testing.T) {
	for _, tc := range loadReasoningSelectionCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			ref := referenceSelect(t, tc.raw)
			req := nativeChatRequest(t, `{"model":"m","messages":[`+tc.raw+`]}`)
			require.Len(t, req.Messages, 1)
			assistant := req.Messages[0].ChatAssistantMessage

			if ref.selection == nil && !ref.sentDetailsArray {
				// Nothing to select and no caller details: the shared decoder is
				// free to leave the assistant field nil, which is what it does.
				if assistant != nil {
					assert.Nil(t, assistant.Reasoning,
						"an all-empty reasoning input must select nothing")
					assert.Empty(t, assistant.ReasoningDetails,
						"an all-empty reasoning input must keep no details")
				}
				return
			}

			require.NotNil(t, assistant, "native ingress must retain the message's reasoning fields")
			if ref.selection == nil {
				assert.Nil(t, assistant.Reasoning,
					"native ingress must select nothing when no spelling carries text")
			} else {
				require.NotNil(t, assistant.Reasoning,
					"native ingress must select replayed reasoning")
				assert.Equal(t, *ref.selection, *assistant.Reasoning,
					"native ingress selected the wrong reasoning text")
			}

			if ref.sentDetailsArray {
				// The caller supplied details; they must survive unchanged.
				assertReasoningDetails(t, ref.callerDetails, assistant.ReasoningDetails, tc.name)
				return
			}

			// The caller sent no reasoning_details array, so the decoder synthesized
			// one. It must carry the value that actually won, not the losing spelling
			// the decoder saw before normalization.
			if ref.selection == nil {
				assert.Empty(t, assistant.ReasoningDetails,
					"no selection means there is nothing to synthesize")
				return
			}
			require.Len(t, assistant.ReasoningDetails, 1,
				"exactly one detail must be synthesized for the selected value")
			// Assert the whole entry, not just its text: a synthesized detail that
			// changes type or index still carries the right text.
			assert.Equal(t, textDetailExpect(0, *ref.selection), reasoningDetailExpect{
				index:     assistant.ReasoningDetails[0].Index,
				typ:       assistant.ReasoningDetails[0].Type,
				text:      assistant.ReasoningDetails[0].Text,
				summary:   assistant.ReasoningDetails[0].Summary,
				signature: assistant.ReasoningDetails[0].Signature,
				data:      assistant.ReasoningDetails[0].Data,
			}, "the synthesized detail must be a reasoning.text entry at index 0")
		})
	}
}

// TestNativeChatIngressReasoningSelectionFixtureIntegrity guards the shared
// fixture. The loader already rejects an empty file, a reordered or truncated row
// set, a duplicated input, and a row whose input stops exercising its named
// scenario; this test names those guarantees and checks that exactly the rows
// which must select nothing do so.
func TestNativeChatIngressReasoningSelectionFixtureIntegrity(t *testing.T) {
	cases := loadReasoningSelectionCases(t)
	require.Len(t, cases, len(reasoningSelectionCaseNames))
	for _, tc := range cases {
		assert.NotEmpty(t, tc.raw, "row %q must carry an input", tc.name)
	}
	assertReferenceSelectsNothing(t)
}

// TestNativeChatIngressReasoningDetailsReachOpenAIWire drives the real native
// ingress seam and then the provider request conversion, asserting that a request
// whose only reasoning spelling was reasoning_details reaches the OpenAI wire as
// non-empty reasoning_content and never leaks the ingress-only reasoning_details.
func TestNativeChatIngressReasoningDetailsReachOpenAIWire(t *testing.T) {
	const reasoning = "RD-MARK"
	req := nativeChatRequest(t, `{"model":"deepseek-v4.1-flash","messages":[{"role":"assistant","reasoning_details":[{"index":0,"type":"reasoning.text","text":"RD-MARK"}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}`)

	ctx, cancel := schemas.NewBifrostContextWithCancel(nil)
	defer cancel()
	bifrostReq := &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "deepseek-v4.1-flash",
		Input:    req.Messages,
	}
	wireBody, err := sonic.Marshal(openai.ToOpenAIChatRequest(ctx, bifrostReq))
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, sonic.Unmarshal(wireBody, &decoded))
	messages, ok := decoded["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 1)
	assistant, ok := messages[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, reasoning, assistant["reasoning_content"],
		"the native ingress reasoning marker must be replayed to the custom OpenAI provider")
	require.NotContains(t, assistant, "reasoning_details",
		"reasoning_details is ingress metadata; the provider wire uses reasoning_content")
}

// TestNativeChatRequestSynthesizesReasoningForToolCalls pins the placeholder
// that keeps a thinking-mode upstream replayable. That upstream rejects an
// assistant tool-call turn that replays no reasoning at all, and rejects an
// empty replay too, so a turn whose real reasoning is unavailable still carries
// non-empty reasoning. Plain assistant turns, callers that did supply reasoning,
// and routes that were not measured to enforce the contract stay untouched.
func TestNativeChatRequestSynthesizesReasoningForToolCalls(t *testing.T) {
	const toolCall = `{"id":"call_1","type":"function","function":{"name":"ping","arguments":"{}"}}`

	t.Run("deepseek tool-call turn gains a placeholder", func(t *testing.T) {
		req := nativeChatRequest(t, `{"model":"opencode-go-custom/deepseek-go","messages":[
			{"role":"user","content":"call ping"},
			{"role":"assistant","tool_calls":[`+toolCall+`]},
			{"role":"tool","tool_call_id":"call_1","content":"pong"}]}`)

		assistant := req.Messages[1].ChatAssistantMessage
		require.NotNil(t, assistant)
		require.NotNil(t, assistant.Reasoning,
			"an assistant tool-call turn must replay non-empty reasoning")
		assert.Equal(t, reasoningReplayPlaceholder, *assistant.Reasoning)
		require.Len(t, assistant.ToolCalls, 1, "the tool call itself must survive")
	})

	t.Run("plain assistant turn stays untouched", func(t *testing.T) {
		req := nativeChatRequest(t, `{"model":"opencode-go-custom/deepseek-go","messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":"hello"}]}`)

		assistant := req.Messages[1].ChatAssistantMessage
		if assistant != nil {
			assert.Nil(t, assistant.Reasoning,
				"a turn without tool calls does not need a replay")
		}
	})

	t.Run("other routes stay untouched", func(t *testing.T) {
		req := nativeChatRequest(t, `{"model":"opencode-go-custom/glm-go","messages":[
			{"role":"user","content":"call ping"},
			{"role":"assistant","tool_calls":[`+toolCall+`]},
			{"role":"tool","tool_call_id":"call_1","content":"pong"}]}`)

		assistant := req.Messages[1].ChatAssistantMessage
		require.NotNil(t, assistant)
		assert.Nil(t, assistant.Reasoning,
			"only measured DeepSeek routes replay a placeholder")
	})

	t.Run("caller reasoning is never replaced", func(t *testing.T) {
		req := nativeChatRequest(t, `{"model":"opencode-go-custom/deepseek-go","messages":[
			{"role":"assistant","reasoning_content":"real","tool_calls":[`+toolCall+`]}]}`)

		assistant := req.Messages[0].ChatAssistantMessage
		require.NotNil(t, assistant)
		require.NotNil(t, assistant.Reasoning)
		assert.Equal(t, "real", *assistant.Reasoning)
	})
}
