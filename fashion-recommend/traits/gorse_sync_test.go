package traits

import (
	"regexp"
	"testing"

	"fashion-recommend/database"

	"github.com/stretchr/testify/assert"
)

// newTestSync 构造一个不连 Gorse、不连数据库的 GorseSync。
// 本文件只覆盖 convertTraitsToLabels，它是纯函数，两个依赖都用不上。
func newTestSync() *GorseSync {
	return &GorseSync{
		minScore:     defaultMinScore,
		maxPerPrefix: defaultMaxPerPrefix,
	}
}

// scoreSuffix 匹配 "…:0.8" 这种拖在末尾的浮点数。
var scoreSuffix = regexp.MustCompile(`:\d+(\.\d+)?$`)

// TestNoFloatSuffix 是本次修复的回归锚点。
//
// 分数拼进 label 会让同一个偏好按小数点后一位裂成互不相同的字符串，每个都是
// 单例，于是全部被 Gorse 的「第二次出现才入索引」过滤掉 —— 线上
// NumUserLabels=0 就是这么来的。任何前缀都不允许再带浮点后缀。
func TestNoFloatSuffix(t *testing.T) {
	labels := newTestSync().convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"minimalist": 0.8, "casual": 0.75},
		ColorPreferences: map[string]float64{"black": 0.9},
		PriceSensitivity: "low",
		BrandPreferences: []string{"uniqlo"},
		Occasions:        []string{"work"},
		Interests:        []string{"sustainable"},
		Keywords:         []string{"oversize"},
	})

	assert.NotEmpty(t, labels)
	for _, label := range labels {
		assert.NotRegexp(t, scoreSuffix, label,
			"label %q 带了分数后缀，会被 Gorse 的频次过滤丢弃", label)
	}
}

// TestScoresGateOnly 证明分数只影响「收不收」，不影响 label 长什么样：
// 同一个风格给两个不同的分数，只要都过阈值，产出的字符串必须完全相同。
func TestScoresGateOnly(t *testing.T) {
	g := newTestSync()

	high := g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"minimalist": 0.9},
	})
	low := g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"minimalist": 0.4},
	})

	assert.Equal(t, []string{"style:minimalist"}, high)
	assert.Equal(t, high, low, "两个用户的同一偏好必须产出同一字符串，否则无法跨用户共现")
}

// TestThresholdBoundary 锁住阈值语义：>= 收，< 丢。
//
// 边界选在 0.4，因为那是 extractor 纯关键词路径的天花板
// （normalizeScores 后 top-1 = 1.0，再乘 mergeTraits 的 keywordWeight 0.4）。
// 原来的 > 0.5 高于这个天花板，等于把全部纯关键词抽取结果静默丢光。
func TestThresholdBoundary(t *testing.T) {
	g := newTestSync()

	assert.Equal(t, []string{"style:formal"}, g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"formal": 0.4},
	}), "关键词路径天花板 0.4 必须能过阈值")

	assert.Empty(t, g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"formal": 0.34},
	}), "低于阈值必须丢弃")

	assert.Equal(t, []string{"style:formal"}, g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"formal": defaultMinScore},
	}), "恰好等于阈值按收处理")
}

// TestPriceRangeNamespace 覆盖 A3：前缀和取值都要落到 item 侧的词表上。
func TestPriceRangeNamespace(t *testing.T) {
	g := newTestSync()

	for trait, want := range map[string]string{
		"low":    "price_range:budget",
		"medium": "price_range:mid",
		"high":   "price_range:premium",
	} {
		assert.Equal(t, []string{want}, g.convertTraitsToLabels(&database.TraitsData{
			PriceSensitivity: trait,
		}), "price_sensitivity=%s", trait)
	}

	// 词表外的取值宁可不发，也不要发一个 item 侧不存在的字符串
	assert.Empty(t, g.convertTraitsToLabels(&database.TraitsData{
		PriceSensitivity: "mid-range",
	}))
	assert.Empty(t, g.convertTraitsToLabels(&database.TraitsData{
		PriceSensitivity: "",
	}))
}

// TestDeterministicOrder 防止 map 遍历顺序泄漏到写给 Gorse 的数组里 ——
// 同一份特质同步两次，存储层看到的东西必须一样。
func TestDeterministicOrder(t *testing.T) {
	g := newTestSync()
	traits := &database.TraitsData{
		StylePreferences: map[string]float64{
			"minimalist": 0.9, "casual": 0.9, "formal": 0.8, "sporty": 0.7,
		},
		ColorPreferences: map[string]float64{"black": 0.6, "white": 0.6},
	}

	first := g.convertTraitsToLabels(traits)
	for i := 0; i < 20; i++ {
		assert.Equal(t, first, g.convertTraitsToLabels(traits))
	}
	// 同分按名字升序，高分在前
	assert.Equal(t, []string{"style:casual", "style:minimalist", "style:formal", "style:sporty"},
		first[:4])
}

// TestMaxPerPrefix 锁住每前缀上限。上限存在是为了让消融实验的两条臂
// （LLM trait vs 聚合 item label）的标签数量不成为混淆项。
func TestMaxPerPrefix(t *testing.T) {
	g := newTestSync()
	scores := map[string]float64{}
	for _, s := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		scores[s] = 0.9
	}

	labels := g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: scores,
		Keywords:         []string{"k1", "k2", "k3", "k4", "k5", "k6", "k7"},
	})

	var styles, keywords int
	for _, l := range labels {
		switch {
		case len(l) > 6 && l[:6] == "style:":
			styles++
		case len(l) > 8 && l[:8] == "keyword:":
			keywords++
		}
	}
	assert.Equal(t, defaultMaxPerPrefix, styles)
	assert.Equal(t, defaultMaxPerPrefix, keywords)
}

// TestSharedLabelAcrossUsers 是 A6 的单元版本：两个分数不同的用户，
// 只要偏好相同就必须产出同一个字符串 —— 这是 Gorse 肯把 label 入索引的
// 唯一前提（第二次出现才收）。修复前这里会得到两个不同的字符串。
func TestSharedLabelAcrossUsers(t *testing.T) {
	g := newTestSync()

	userA := g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"minimalist": 0.8},
		ColorPreferences: map[string]float64{"white": 0.55},
	})
	userB := g.convertTraitsToLabels(&database.TraitsData{
		StylePreferences: map[string]float64{"minimalist": 0.42},
		ColorPreferences: map[string]float64{"white": 0.91},
	})

	shared := map[string]bool{}
	for _, a := range userA {
		for _, b := range userB {
			if a == b {
				shared[a] = true
			}
		}
	}
	assert.True(t, shared["style:minimalist"])
	assert.True(t, shared["color:white"])
}

// TestEmptyTraits 空特质不应产出任何 label（也不应 panic）。
func TestEmptyTraits(t *testing.T) {
	assert.Empty(t, newTestSync().convertTraitsToLabels(&database.TraitsData{}))
}
