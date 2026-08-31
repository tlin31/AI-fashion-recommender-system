package traits

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"

	"fashion-recommend/client"
	"fashion-recommend/database"
	"fashion-recommend/models"
)

const (
	// defaultMinScore 是 style/color 偏好进入 Gorse label 的门槛。
	//
	// 0.35 不是拍脑袋：extractor 的 normalizeScores 先把关键词路径的分数除以
	// max（top-1 恒为 1.0），随后 mergeTraits 按 keyword*0.4 + ai*0.6 合并。
	// 所以 AI 那条腿缺失或失败时，合并分的天花板就是 0.4。原来的 > 0.5 高于
	// 这个天花板，等于把全部纯关键词抽取结果静默丢光。0.35 卡在 0.4 之下、
	// 第二梯队（关键词命中一次 = 0.3 归一化后更低）之上。
	defaultMinScore = 0.35

	// defaultMaxPerPrefix 限制每个前缀最多产出几个 label。
	//
	// 存在的理由是消融实验：对比「LLM 抽取的 trait」和「聚合 item label」两条臂
	// 时，如果一条臂天然产出的 label 更多，NDCG 的差异就分不清是信号质量还是
	// 标签数量。上限把这个混淆项钉死。
	defaultMaxPerPrefix = 5
)

// priceRangeVocab 把 trait 的 price_sensitivity 取值映射到 item 侧实际使用的
// price_range 词表。
//
// rag_products.price_range 的取值是 budget/mid/premium（外加 seed 时补的
// unknown），而 extractor 产出的是 low/medium/high。只把前缀从 price: 改成
// price_range: 而不动取值，两侧字符串依然永远对不上。
var priceRangeVocab = map[string]string{
	"low":    "budget",
	"medium": "mid",
	"high":   "premium",
}

// GorseSync Gorse 同步服务
type GorseSync struct {
	gorseClient *client.GorseClient
	db          *database.DB

	// minScore / maxPerPrefix 见上方常量注释。做成字段而不是常量，是为了让
	// 消融实验能用 env 扫阈值，不必改代码重新编译。
	minScore     float64
	maxPerPrefix int
}

// NewGorseSync 创建 Gorse 同步服务
func NewGorseSync(gorseClient *client.GorseClient, db *database.DB) *GorseSync {
	return &GorseSync{
		gorseClient:  gorseClient,
		db:           db,
		minScore:     envFloat("TRAIT_LABEL_MIN_SCORE", defaultMinScore),
		maxPerPrefix: envInt("TRAIT_LABEL_MAX_PER_PREFIX", defaultMaxPerPrefix),
	}
}

func envFloat(key string, defaultValue float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil {
		return v
	}
	return defaultValue
}

func envInt(key string, defaultValue int) int {
	if v, err := strconv.Atoi(os.Getenv(key)); err == nil && v > 0 {
		return v
	}
	return defaultValue
}

// SyncUserTraitsToGorse 同步用户特质到 Gorse
func (g *GorseSync) SyncUserTraitsToGorse(userID string) error {
	// 1. 从数据库获取用户特质
	userTraits, err := g.db.GetUserTraits(userID)
	if err != nil {
		return fmt.Errorf("获取用户特质失败: %w", err)
	}

	if userTraits == nil {
		// 用户还没有特质数据
		return nil
	}

	// 2. 解析特质数据
	var traits database.TraitsData
	if err := json.Unmarshal(userTraits.Traits, &traits); err != nil {
		return fmt.Errorf("解析特质数据失败: %w", err)
	}

	// 3. 转换为 Gorse Labels
	labels := g.convertTraitsToLabels(&traits)

	// 4. 更新 Gorse 用户
	user := models.User{
		UserId:  userID,
		Labels:  labels,
		Comment: fmt.Sprintf("特质置信度: %.2f", userTraits.ConfidenceScore),
	}

	if err := g.gorseClient.InsertUser(user); err != nil {
		return fmt.Errorf("更新 Gorse 用户失败: %w", err)
	}

	return nil
}

// convertTraitsToLabels 将特质转换为 Gorse Labels。
//
// 分数只做门控，绝不进字符串。Gorse 的 CTR 数据集只在一个 label 第二次出现
// （跨用户）时才把它加进索引，见 master/tasks.go 的 userLabelCount 逻辑；
// 把分数拼进 label（"style:minimalist:0.8"）会让同一个偏好因为小数点后一位
// 的差异裂成互不相同的字符串，于是每个都是单例、每个都被丢弃。这正是线上
// NumUserLabels=0 而 NumItemLabels=15,332 的原因 —— item 侧的 label 一直是
// 干净的 "prefix:value"。
func (g *GorseSync) convertTraitsToLabels(traits *database.TraitsData) []string {
	labels := []string{}

	// 风格 / 颜色偏好：过阈值，按分数取前 maxPerPrefix 个
	labels = append(labels, g.topScored("style", traits.StylePreferences)...)
	labels = append(labels, g.topScored("color", traits.ColorPreferences)...)

	// 价格敏感度：映射到 item 侧的 price_range 词表，对不上的取值直接丢弃，
	// 而不是原样透传出一个 item 侧不存在的字符串
	if v, ok := priceRangeVocab[traits.PriceSensitivity]; ok {
		labels = append(labels, "price_range:"+v)
	}

	// 无分数的列表型特质：只截断，不排序（原始顺序本身就是抽取器给的优先级）
	labels = append(labels, g.capped("brand", traits.BrandPreferences)...)
	labels = append(labels, g.capped("occasion", traits.Occasions)...)
	labels = append(labels, g.capped("interest", traits.Interests)...)
	labels = append(labels, g.capped("keyword", traits.Keywords)...)

	return labels
}

// topScored 过滤掉低于阈值的偏好，按分数降序取前 maxPerPrefix 个。
//
// 排序还有一个副作用是必要的：Go 的 map 遍历顺序随机，原实现每次同步都会给
// Gorse 写入顺序不同的 label 数组，同一份特质的两次同步在存储层看起来不一样。
func (g *GorseSync) topScored(prefix string, scores map[string]float64) []string {
	type scored struct {
		name  string
		score float64
	}
	kept := make([]scored, 0, len(scores))
	for name, score := range scores {
		if score >= g.minScore {
			kept = append(kept, scored{name: name, score: score})
		}
	}
	sort.Slice(kept, func(i, j int) bool {
		if kept[i].score != kept[j].score {
			return kept[i].score > kept[j].score
		}
		return kept[i].name < kept[j].name // 同分时按名字定序，保证可重现
	})
	if len(kept) > g.maxPerPrefix {
		kept = kept[:g.maxPerPrefix]
	}
	labels := make([]string, 0, len(kept))
	for _, s := range kept {
		labels = append(labels, prefix+":"+s.name)
	}
	return labels
}

// capped 给列表型特质加前缀并截断到 maxPerPrefix。
func (g *GorseSync) capped(prefix string, values []string) []string {
	if len(values) > g.maxPerPrefix {
		values = values[:g.maxPerPrefix]
	}
	labels := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" {
			continue
		}
		labels = append(labels, prefix+":"+v)
	}
	return labels
}

// GenerateImplicitFeedback 根据对话生成隐式反馈
func (g *GorseSync) GenerateImplicitFeedback(userID string, traits *database.TraitsData) error {
	// 这里可以根据用户特质生成一些隐式的"喜欢"反馈
	// 例如：如果用户喜欢简约风格，可以给简约风格的商品添加隐式反馈
	
	// 暂时不实现，留作后续优化
	return nil
}
