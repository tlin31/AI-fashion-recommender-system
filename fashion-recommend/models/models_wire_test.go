package models

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// 这些用例断言的是**字面 JSON key**，不是 struct 往返。
//
// 往返测试对 tag 错误完全免疫：同一个 struct 编码再解码，就算 tag 写成
// "banana" 也能对上。Gorse 那头是另一个 struct，所以只有字面 key 才算数。
// 真实的 Gorse 线格式（curl 验证过）：
//
//	POST /api/user  {"UserId":"u1","Labels":["style:minimalist"],"Comment":"..."}

func TestUserWireFormat(t *testing.T) {
	raw, err := json.Marshal(User{
		UserId:  "u1",
		Labels:  []string{"style:minimalist"},
		Comment: "hi",
	})
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(raw, &got))

	assert.Equal(t, "u1", got["UserId"], "Gorse 认的是 UserId，不是 user_id")
	assert.Contains(t, got, "Labels")
	assert.NotContains(t, got, "user_id")
	assert.NotContains(t, got, "labels")
}

func TestItemWireFormat(t *testing.T) {
	raw, err := json.Marshal(Item{ItemId: "i1", Labels: []string{"style:sporty"}})
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(raw, &got))

	assert.Equal(t, "i1", got["ItemId"])
	assert.Contains(t, got, "Labels")
}

// TestUserBuilderNamespaces 锁住 user 侧和 item 侧共用同一套前缀 ——
// price_range: 而不是 price_preference:，brand: 而不是 favorite_brand:。
func TestUserBuilderNamespaces(t *testing.T) {
	user := NewFashionUser("u1").
		WithStyle("minimalist").
		WithPricePreference("budget").
		WithFavoriteBrands("uniqlo").
		Build()

	assert.Contains(t, user.Labels, "style:minimalist")
	assert.Contains(t, user.Labels, "price_range:budget")
	assert.Contains(t, user.Labels, "brand:uniqlo")

	for _, label := range user.Labels {
		assert.NotContains(t, label, "price_preference:")
		assert.NotContains(t, label, "favorite_brand:")
	}
}
