package config

import (
	"testing"
	"time"

	"github.com/expr-lang/expr"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fashion-recommend/config/config.toml 里的每个 expr 表达式，都要**编译并求值**
// 一遍。这个测试住在 gorse 根模块而不是 fashion-recommend 里，因为 expr 的
// 求值环境（哪个变量被绑定、duration() 支持哪些单位）是由这边的 logics 包定义
// 的 —— fashion-recommend 不依赖 gorse，把断言放那边就等于凭记忆抄一份语义。
//
// 为什么必须**求值**而不只是编译：这类 bug 已经出现三次，其中两次编译是过的。
//
//	column = "Labels"                             → 编译期失败（未定义标识符）
//	duration('30d')                               → 编译通过，**求值时**才失败
//	filter = "item.Timestamp > now() - ..."        → 同上
//
// 只做编译检查会漏掉后两个。而运行时它们的表现是每个 item 刷一行错误日志、
// dashboard 依然把任务报成 Complete —— 静默到把 Docker 磁盘写满才被发现。
const fashionConfigPath = "../fashion-recommend/config/config.toml"

// fashionSampleItem 的 Labels 用的是 map 形式，因为 data.Item.Labels 的类型
// 是 any，而 seeder 现在写进去的就是 map（见 models.ItemLabels）。
//
// 这里刻意用 []any 而不是 []string：Labels 从 Postgres 出来要经过
// gorm 的 json serializer 反序列化进一个 any，结果必然是
// map[string]any{"f": []any{...}}。写成 []string 会让测试跑在一个生产中
// 不存在的类型上。
func fashionSampleItem() data.Item {
	return data.Item{
		ItemId:    "B00006XXGO",
		Timestamp: time.Now().Add(-24 * time.Hour),
		Labels: map[string]any{
			"f":           []any{"type:t-shirt", "cat:tops", "style:minimalist", "color:white"},
			"brand":       "uniqlo",
			"price_range": "mid",
			"avg_rating":  "4.5",
		},
		Categories: []string{"fashion"},
	}
}

// fashionLegacyFlatItem 是重新 seed 之前的形状。留着不是为了兼容，而是为了
// 把「seed 没跑完」这个故障形态钉在测试里 —— 见 TestFashionColumnRejectsFlatLabels。
func fashionLegacyFlatItem() data.Item {
	item := fashionSampleItem()
	item.Labels = []any{"style:minimalist", "color:white"}
	return item
}

func fashionSampleFeedback() []data.Feedback {
	return []data.Feedback{{
		FeedbackKey: data.FeedbackKey{
			FeedbackType: "purchase",
			UserId:       "u1",
			ItemId:       "B00006XXGO",
		},
		Timestamp: time.Now().Add(-2 * time.Hour),
	}}
}

func TestFashionConfigLoads(t *testing.T) {
	cfg, err := LoadConfig(fashionConfigPath)
	require.NoError(t, err)
	require.NoError(t, cfg.Validate())
}

func TestFashionNonPersonalizedExpressions(t *testing.T) {
	cfg, err := LoadConfig(fashionConfigPath)
	require.NoError(t, err)

	// 环境与 logics/non_personalized.go:47,62 保持一致
	scoreEnv := map[string]any{"item": data.Item{}, "feedback": []data.Feedback{}}
	filterEnv := map[string]any{"item": data.Item{}}
	runEnv := map[string]any{"item": fashionSampleItem(), "feedback": fashionSampleFeedback()}

	for _, c := range cfg.Recommend.NonPersonalized {
		t.Run(c.Name, func(t *testing.T) {
			program, err := expr.Compile(c.Score, expr.Env(scoreEnv))
			require.NoError(t, err, "score 编译失败")
			_, err = expr.Run(program, runEnv)
			require.NoError(t, err, "score 求值失败 —— 编译通过不代表能跑，"+
				"duration() 只认 ns/us/ms/s/m/h，没有 d")

			if c.Filter == "" {
				return
			}
			program, err = expr.Compile(c.Filter, expr.Env(filterEnv))
			require.NoError(t, err, "filter 编译失败")
			_, err = expr.Run(program, runEnv)
			require.NoError(t, err, "filter 求值失败")
		})
	}
}

// TestFashionSimilarityColumns 覆盖两个 tags 相似度臂的 column 表达式。
//
// 注意两侧绑定的变量不同：item-to-item 绑 item（logics/item_to_item.go:199），
// user-to-user 绑 user（logics/user_to_user.go:112,167）。把 item 侧的
// "item.Labels" 照抄到 user-to-user 编译不过，反之亦然 —— 这正是原来两处
// 都写成裸 "Labels" 时没人发现的原因：错得一样，所以看起来一致。
// TestFashionColumnRejectsFlatLabels 断言 column = "item.Labels.f" 碰上扁平
// 标签时是**运行时**失败，不是编译失败。
//
// 这个区别是 Day 2 那三个 bug 的核心教训在标签重构上的重演：编译期检查在这里
// 什么都抓不到（表达式本身合法，item.Labels 的静态类型是 any），错误只在每个
// item 求值时才出现 —— 而那种错误 Gorse 会一个 item 刷一行日志，把任务照常
// 报成 Complete。所以重新 seed 只完成一半的后果不是「一半商品没有邻居」，
// 而是整个任务在日志里安静地失败。
func TestFashionColumnRejectsFlatLabels(t *testing.T) {
	program, err := expr.Compile(`item.Labels.f`,
		expr.Env(map[string]any{"item": data.Item{}}))
	require.NoError(t, err, "编译期看不出问题 —— 这正是重点")

	_, err = expr.Run(program, map[string]any{"item": fashionLegacyFlatItem()})
	require.Error(t, err, "扁平标签必须在求值时失败，否则这条保护形同虚设")

	_, err = expr.Run(program, map[string]any{"item": fashionSampleItem()})
	require.NoError(t, err)
}

func TestFashionSimilarityColumns(t *testing.T) {
	cfg, err := LoadConfig(fashionConfigPath)
	require.NoError(t, err)

	for _, c := range cfg.Recommend.ItemToItem {
		if c.Column == "" {
			continue
		}
		t.Run("item-to-item/"+c.Name, func(t *testing.T) {
			program, err := expr.Compile(c.Column, expr.Env(map[string]any{"item": data.Item{}}))
			require.NoError(t, err)
			_, err = expr.Run(program, map[string]any{"item": fashionSampleItem()})
			require.NoError(t, err)
		})
	}

	for _, c := range cfg.Recommend.UserToUser {
		if c.Column == "" {
			continue
		}
		t.Run("user-to-user/"+c.Name, func(t *testing.T) {
			program, err := expr.Compile(c.Column, expr.Env(map[string]any{"user": data.User{}}))
			require.NoError(t, err)
			_, err = expr.Run(program, map[string]any{
				"user": data.User{UserId: "u1", Labels: []string{"style:minimalist"}},
			})
			require.NoError(t, err)
		})
	}
}

// TestFashionNoBareLabelsColumn 是对原 bug 的直接锚点。
func TestFashionNoBareLabelsColumn(t *testing.T) {
	cfg, err := LoadConfig(fashionConfigPath)
	require.NoError(t, err)

	for _, c := range cfg.Recommend.ItemToItem {
		assert.NotEqual(t, "Labels", c.Column,
			"item-to-item %s: column 是 expr 表达式不是列名，应为 item.Labels", c.Name)
	}
	for _, c := range cfg.Recommend.UserToUser {
		assert.NotEqual(t, "Labels", c.Column,
			"user-to-user %s: 应为 user.Labels（不是 item.Labels）", c.Name)
	}
}
