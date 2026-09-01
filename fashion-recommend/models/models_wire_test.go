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
	raw, err := json.Marshal(Item{
		ItemId: "i1",
		Labels: ItemLabels{
			Features:   []string{"type:t-shirt", "cat:tops", "style:sporty"},
			Brand:      "zara",
			PriceRange: "mid",
			AvgRating:  "4.5",
		},
	})
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(raw, &got))
	assert.Equal(t, "i1", got["ItemId"])

	labels, ok := got["Labels"].(map[string]any)
	require.True(t, ok, "Labels 必须是 map，不是数组 —— tags item-to-item 靠 "+
		"expr 表达式选分支，扁平数组下 column = \"item.Labels.f\" 无法求值")

	// 这个 key 的名字是配置里的字面量。config.toml 写的是
	// column = "item.Labels.f"，改这个字符串就等于让 item-to-item 静默失效
	// —— 而 Day 2 已经证明过，expr 求值失败时 dashboard 照样报 Complete。
	assert.Contains(t, labels, "f", `相似度分支的 key 必须叫 "f"`)
	assert.Equal(t,
		[]any{"type:t-shirt", "cat:tops", "style:sporty"}, labels["f"])

	// 载体和粗粒度属性必须在 f 之外：它们要留给 CTR，但不能进相似度。
	assert.Equal(t, "zara", labels["brand"])
	assert.Equal(t, "mid", labels["price_range"])
	assert.Equal(t, "4.5", labels["avg_rating"],
		"avg_rating 必须是字符串 —— map 键下的 JSON 数值 ctr.convertLabels "+
			"和 flatten 都读不到，会得到一个看起来像特征的惰性字段")
}

// TestItemLabelsAcceptsFlatArray 覆盖重新 seed 期间的混合状态：Gorse 里同时
// 存在两种形式的 item，解码器只认 map 的话 API 会对着一半商品报错。
func TestItemLabelsAcceptsFlatArray(t *testing.T) {
	var item Item
	require.NoError(t, json.Unmarshal(
		[]byte(`{"ItemId":"i1","Labels":["style:sporty","color:black"]}`), &item))
	assert.Equal(t, []string{"style:sporty", "color:black"}, item.Labels.Features)
	assert.Empty(t, item.Labels.Brand)
}

// TestItemCarrierRoundTrip 锁住载体的落点。name/price 曾经是标签，那是放错了
// 字段；它们现在走 Comment，因为 logics/ master/ model/ dataset/ 里没有任何
// 代码读 .Comment。
func TestItemCarrierRoundTrip(t *testing.T) {
	price := 12.34
	raw, err := json.Marshal(ItemCarrier{Name: "Linen Shirt", Price: &price, Desc: "d"})
	require.NoError(t, err)

	got := ParseItemCarrier(string(raw))
	assert.Equal(t, "Linen Shirt", got.Name)
	require.NotNil(t, got.Price)
	assert.InDelta(t, 12.34, *got.Price, 1e-9)

	// 价格缺失和价格为 0 必须可区分 —— 指针类型就是为这个。
	noPrice := ParseItemCarrier(`{"name":"x","price":null}`)
	assert.Nil(t, noPrice.Price)

	// 旧数据：Comment 存的是纯描述文本，不是 JSON。不能当成错误。
	legacy := ParseItemCarrier("just a plain description")
	assert.Equal(t, "just a plain description", legacy.Desc)
	assert.Empty(t, legacy.Name)
}

// TestItemBuilderSplitsFeaturesFromAttributes 锁住 builder 的分流：
// style/color/material/occasion 进 f，brand/price_range 不进。
func TestItemBuilderSplitsFeaturesFromAttributes(t *testing.T) {
	item := NewFashionItem("i1").
		WithStyle("minimalist").
		WithColor("black").
		WithBrand("uniqlo").
		WithPriceRange("budget").
		Build()

	assert.Contains(t, item.Labels.Features, "style:minimalist")
	assert.Contains(t, item.Labels.Features, "color:black")
	assert.Equal(t, "uniqlo", item.Labels.Brand)
	assert.Equal(t, "budget", item.Labels.PriceRange)

	for _, f := range item.Labels.Features {
		assert.NotContains(t, f, "brand:",
			"brand 近乎唯一，进 f 就是给相似度加一个永不匹配却拉高范数的标签")
		assert.NotContains(t, f, "price_range:",
			"price_range 约 80% 是 mid，进 f 就是给每一对商品加同一个共享项")
	}
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
