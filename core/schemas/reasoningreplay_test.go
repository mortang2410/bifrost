package schemas

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// reasoningReplayCase is one row of a resolver table. record is the datasheet value,
// installed through the capability resolver so the lookup takes the production path.
type reasoningReplayCase struct {
	name   string
	query  string
	custom *CustomProviderConfig
	record string
	want   string
}

// replayCaps resolves the capability record for model with its reasoning replay field
// set to field. An empty field still installs a record, which reads as unset.
func replayCaps(t *testing.T, model, field string) ModelCaps {
	t.Helper()
	setCapabilityOverride(t, model, ModelCapabilities{ReasoningReplayField: field})
	return ResolveModelCaps(Anthropic, model)
}

// recordCaps returns a ModelCaps carrying a datasheet record with the given field. It
// is used where the request model is empty: ResolveModelCaps returns no record for an
// empty model, so the resolver is handed a record directly.
func recordCaps(field string) ModelCaps {
	return ModelCaps{record: &ModelCapabilities{ReasoningReplayField: field}}
}

// Precedence is first hit wins: per-model entry, provider default, datasheet record,
// then auto. Each layer holds a different value, so a swapped layer order returns a
// different answer and fails the row that exposes it.
func TestResolveReasoningReplayFieldPrecedence(t *testing.T) {
	const model = "model-a"
	tests := []reasoningReplayCase{
		{
			name:  "per-model entry beats provider default and record",
			query: model,
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: ReasoningReplayReasoningText},
				ReasoningReplayField:  ReasoningReplayReasoningContent,
			},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningText,
		},
		{
			name:  "provider default beats record when the model has no entry",
			query: "model-b",
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: ReasoningReplayReasoningText},
				ReasoningReplayField:  ReasoningReplayReasoningContent,
			},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningContent,
		},
		{
			name:   "record answers when the config names nothing",
			query:  model,
			custom: &CustomProviderConfig{},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningDetails,
		},
		{
			name:   "auto when no source names a value",
			query:  model,
			custom: &CustomProviderConfig{},
			record: "",
			want:   ReasoningReplayAuto,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := replayCaps(t, tt.query, tt.record)
			assert.Equal(t, tt.want, ResolveReasoningReplayField(tt.query, tt.custom, caps))
		})
	}
}

// An explicit auto is a valid answer, so it stops the search. A lower-priority source
// that names a real value must not replace it.
func TestResolveReasoningReplayFieldExplicitAutoStops(t *testing.T) {
	const model = "model-a"
	tests := []reasoningReplayCase{
		{
			name:  "per-model auto beats provider default and record",
			query: model,
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: ReasoningReplayAuto},
				ReasoningReplayField:  ReasoningReplayReasoningContent,
			},
			record: ReasoningReplayReasoning,
			want:   ReasoningReplayAuto,
		},
		{
			name:   "provider default auto beats record",
			query:  model,
			custom: &CustomProviderConfig{ReasoningReplayField: ReasoningReplayAuto},
			record: ReasoningReplayReasoning,
			want:   ReasoningReplayAuto,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := replayCaps(t, tt.query, tt.record)
			assert.Equal(t, tt.want, ResolveReasoningReplayField(tt.query, tt.custom, caps))
		})
	}
}

// An unrecognised value resolves to auto and stops the search. A lower-priority source
// must not supply the answer once a higher-priority source has named an unknown value.
func TestResolveReasoningReplayFieldUnknownResolvesToAuto(t *testing.T) {
	const model = "model-a"
	tests := []reasoningReplayCase{
		{
			name:  "unknown per-model value beats provider default and record",
			query: model,
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: "bogus"},
				ReasoningReplayField:  ReasoningReplayReasoningContent,
			},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayAuto,
		},
		{
			name:   "unknown provider default beats record",
			query:  model,
			custom: &CustomProviderConfig{ReasoningReplayField: "bogus"},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayAuto,
		},
		{
			name:   "unknown datasheet record resolves to auto",
			query:  model,
			custom: &CustomProviderConfig{},
			record: "bogus",
			want:   ReasoningReplayAuto,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := replayCaps(t, tt.query, tt.record)
			assert.Equal(t, tt.want, ResolveReasoningReplayField(tt.query, tt.custom, caps))
		})
	}
}

// Matching is exact. A configured value with a surrounding space or different casing is
// unknown, so it resolves to auto rather than to the trimmed or lowercased name. The
// provider default is set to a real value, so a lookup that continued past the unknown
// value would return it instead of auto.
func TestResolveReasoningReplayFieldMatchingIsExact(t *testing.T) {
	const model = "model-a"
	for _, value := range []string{" reasoning", "reasoning ", "Reasoning_Content", "REASONING_TEXT"} {
		t.Run(value, func(t *testing.T) {
			custom := &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: value},
				ReasoningReplayField:  ReasoningReplayReasoningDetails,
			}
			caps := replayCaps(t, model, "")
			assert.Equal(t, ReasoningReplayAuto, ResolveReasoningReplayField(model, custom, caps))
		})
	}
}

// An empty per-model entry means unset for that model, so the resolver moves on to the
// next source.
func TestResolveReasoningReplayFieldEmptyEntryFallsThrough(t *testing.T) {
	const model = "model-a"
	tests := []reasoningReplayCase{
		{
			name:  "empty per-model entry falls through to provider default",
			query: model,
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: ""},
				ReasoningReplayField:  ReasoningReplayReasoningText,
			},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningText,
		},
		{
			name:  "empty per-model entry and empty default fall through to record",
			query: model,
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{model: ""},
			},
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningDetails,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := replayCaps(t, tt.query, tt.record)
			assert.Equal(t, tt.want, ResolveReasoningReplayField(tt.query, tt.custom, caps))
		})
	}
}

// An empty model skips the per-model map. The provider default and the record still
// apply, and a map-only config resolves to auto. Rows pass the record directly because
// ResolveModelCaps returns no record for an empty model.
func TestResolveReasoningReplayFieldEmptyModel(t *testing.T) {
	tests := []struct {
		name   string
		custom *CustomProviderConfig
		caps   ModelCaps
		want   string
	}{
		{
			name:   "map-only config resolves to auto",
			custom: &CustomProviderConfig{ReasoningReplayFields: map[string]string{"model-a": ReasoningReplayReasoningText}},
			caps:   ModelCaps{},
			want:   ReasoningReplayAuto,
		},
		{
			name: "provider default applies",
			custom: &CustomProviderConfig{
				ReasoningReplayFields: map[string]string{"model-a": ReasoningReplayReasoningText},
				ReasoningReplayField:  ReasoningReplayReasoningContent,
			},
			caps: recordCaps(ReasoningReplayReasoningDetails),
			want: ReasoningReplayReasoningContent,
		},
		{
			name:   "record applies when the default is unset",
			custom: &CustomProviderConfig{ReasoningReplayFields: map[string]string{"model-a": ReasoningReplayReasoningText}},
			caps:   recordCaps(ReasoningReplayReasoningDetails),
			want:   ReasoningReplayReasoningDetails,
		},
		{
			name:   "empty-string map key is not a model lookup",
			custom: &CustomProviderConfig{ReasoningReplayFields: map[string]string{"": ReasoningReplayReasoningText}},
			caps:   recordCaps(ReasoningReplayReasoningDetails),
			want:   ReasoningReplayReasoningDetails,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ResolveReasoningReplayField("", tt.custom, tt.caps))
		})
	}
}

// A nil config is valid. The datasheet record answers when it is set, and auto otherwise.
func TestResolveReasoningReplayFieldNilCustom(t *testing.T) {
	tests := []reasoningReplayCase{
		{
			name:   "record answers when present",
			query:  "model-a",
			custom: nil,
			record: ReasoningReplayReasoningDetails,
			want:   ReasoningReplayReasoningDetails,
		},
		{
			name:   "auto when the record is unset",
			query:  "model-a",
			custom: nil,
			record: "",
			want:   ReasoningReplayAuto,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := replayCaps(t, tt.query, tt.record)
			assert.Equal(t, tt.want, ResolveReasoningReplayField(tt.query, tt.custom, caps))
		})
	}

	t.Run("no record at all resolves to auto", func(t *testing.T) {
		assert.Equal(t, ReasoningReplayAuto, ResolveReasoningReplayField("model-a", nil, ModelCaps{}))
	})
}

// Every accepted value comes back verbatim from each of the three sources. none must
// survive as none, not be rewritten to auto.
func TestResolveReasoningReplayFieldAcceptedValuesRoundTrip(t *testing.T) {
	const model = "model-a"
	values := []string{
		ReasoningReplayReasoningContent,
		ReasoningReplayReasoning,
		ReasoningReplayReasoningText,
		ReasoningReplayReasoningDetails,
		ReasoningReplayNone,
		ReasoningReplayAuto,
	}
	for _, value := range values {
		t.Run(value+"/per-model", func(t *testing.T) {
			custom := &CustomProviderConfig{ReasoningReplayFields: map[string]string{model: value}}
			assert.Equal(t, value, ResolveReasoningReplayField(model, custom, ModelCaps{}))
		})
		t.Run(value+"/provider-default", func(t *testing.T) {
			custom := &CustomProviderConfig{ReasoningReplayField: value}
			assert.Equal(t, value, ResolveReasoningReplayField(model, custom, ModelCaps{}))
		})
		t.Run(value+"/datasheet", func(t *testing.T) {
			caps := replayCaps(t, model, value)
			assert.Equal(t, value, ResolveReasoningReplayField(model, nil, caps))
		})
	}
}

// The two config fields use the JSON keys the rest of the configuration uses. Both are
// omitted when empty, including an empty map.
func TestCustomProviderConfigReasoningReplayJSON(t *testing.T) {
	cfg := CustomProviderConfig{
		ReasoningReplayField:  ReasoningReplayReasoningContent,
		ReasoningReplayFields: map[string]string{"model-a": ReasoningReplayReasoningText},
	}
	raw, err := Marshal(cfg)
	require.NoError(t, err)

	var wire map[string]any
	require.NoError(t, Unmarshal(raw, &wire))
	assert.Equal(t, ReasoningReplayReasoningContent, wire["reasoning_replay_field"])
	assert.Equal(t, map[string]any{"model-a": ReasoningReplayReasoningText}, wire["reasoning_replay_fields"])

	var back CustomProviderConfig
	require.NoError(t, Unmarshal(raw, &back))
	assert.Equal(t, cfg.ReasoningReplayField, back.ReasoningReplayField)
	assert.Equal(t, cfg.ReasoningReplayFields, back.ReasoningReplayFields)

	rawEmpty, err := Marshal(CustomProviderConfig{ReasoningReplayFields: map[string]string{}})
	require.NoError(t, err)
	var wireEmpty map[string]any
	require.NoError(t, Unmarshal(rawEmpty, &wireEmpty))
	assert.NotContains(t, wireEmpty, "reasoning_replay_field")
	assert.NotContains(t, wireEmpty, "reasoning_replay_fields")
}

// A datasheet record that sets the field feeds the accessor through the resolver, and
// a record that leaves it empty, or no record at all, returns the caller's fallback.
func TestModelCapsReasoningReplayFieldAccessor(t *testing.T) {
	tests := []struct {
		name string
		wire string
		want string
	}{
		{name: "record field feeds the accessor", wire: `{"reasoning_replay_field":"reasoning_text"}`, want: "reasoning_text"},
		{name: "absent field returns the fallback", wire: `{}`, want: "fallback"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ov ModelCapabilities
			require.NoError(t, Unmarshal([]byte(tt.wire), &ov))
			setCapabilityOverride(t, "model-a", ov)
			assert.Equal(t, tt.want, ResolveModelCaps(Anthropic, "model-a").ReasoningReplayField("fallback"))
		})
	}

	t.Run("no record at all returns the fallback", func(t *testing.T) {
		assert.Equal(t, "fallback", ModelCaps{}.ReasoningReplayField("fallback"))
	})
}
