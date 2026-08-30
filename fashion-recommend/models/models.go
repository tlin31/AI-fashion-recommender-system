package models

import "time"

// User 用户模型。
//
// 字段名必须和 Gorse REST API 的线格式完全一致（UserId / Labels / ...），
// 跟下面的 Item 一样。原来这里是 snake_case 的 user_id / labels，后果是
// POST /api/user(s) 发出去的 user_id 对不上 Gorse 的 UserId 字段 ——
// encoding/json 反序列化对大小写不敏感，所以 labels 侥幸能对上 Labels，
// 但带下划线的 user_id 对不上，于是 Gorse 收到的每个用户 UserId 都是空串，
// 全部写进同一行空 ID 记录。trait 同步因此从来没有真正落到过任何用户身上。
//
// 这个 bug 躲过了 server_test.go，因为那里的 mock Gorse 用同一个 struct
// 编码响应、再用同一个 struct 解码，tag 不一致在往返中被抵消了。参见
// models_wire_test.go —— 那里断言的是字面 JSON key，不是往返。
type User struct {
	UserId    string   `json:"UserId"`
	Labels    []string `json:"Labels"`
	Comment   string   `json:"Comment,omitempty"`
	Subscribe []string `json:"Subscribe,omitempty"`
}

// Item 商品模型
type Item struct {
	ItemId     string    `json:"ItemId"`
	IsHidden   bool      `json:"IsHidden"`
	Categories []string  `json:"Categories"`
	Labels     []string  `json:"Labels"`
	Comment    string    `json:"Comment,omitempty"`
	Timestamp  time.Time `json:"Timestamp"`
}

// Feedback 反馈模型。
//
// 和 User 一样，字段名必须匹配 Gorse 的线格式。storage/data/database.go 的
// Feedback / FeedbackKey 没有 json tag，所以 Gorse 认的是 Go 字段名本身：
// FeedbackType / UserId / ItemId / Value / Timestamp。原来这里的 snake_case
// 带下划线，对不上，反馈写进去时三个 key 全是空的。
type Feedback struct {
	FeedbackType string    `json:"FeedbackType"`
	UserId       string    `json:"UserId"`
	ItemId       string    `json:"ItemId"`
	Value        float64   `json:"Value"`
	Timestamp    time.Time `json:"Timestamp"`
	Comment      string    `json:"Comment,omitempty"`
}

// RecommendRequest 推荐请求
type RecommendRequest struct {
	UserId   string   `json:"user_id"`
	N        int      `json:"n"`
	Category string   `json:"category,omitempty"`
	Offset   int      `json:"offset,omitempty"`
}

// RecommendResponse 推荐响应
type RecommendResponse struct {
	Items []RecommendItem `json:"items"`
	Total int             `json:"total"`
}

// RecommendItem 推荐商品项
type RecommendItem struct {
	ItemId     string   `json:"item_id"`
	Score      float64  `json:"score"`
	Categories []string `json:"categories,omitempty"`
}

// FashionUserBuilder 时尚用户构建器
type FashionUserBuilder struct {
	user User
}

func NewFashionUser(userId string) *FashionUserBuilder {
	return &FashionUserBuilder{
		user: User{
			UserId: userId,
			Labels: make([]string, 0),
		},
	}
}

func (b *FashionUserBuilder) WithGender(gender string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "gender:"+gender)
	return b
}

func (b *FashionUserBuilder) WithAgeGroup(ageGroup string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "age_group:"+ageGroup)
	return b
}

func (b *FashionUserBuilder) WithStyle(styles ...string) *FashionUserBuilder {
	for _, style := range styles {
		b.user.Labels = append(b.user.Labels, "style:"+style)
	}
	return b
}

// WithPricePreference 写入 price_range: 前缀 —— 和 item 侧同名。
// priceRange 应取 item 侧的词表（budget / mid / premium），不是自造的
// mid-range / high-end / luxury 之类。
func (b *FashionUserBuilder) WithPricePreference(priceRange string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "price_range:"+priceRange)
	return b
}

// WithFavoriteBrands 写入 brand: 前缀 —— 和 item 侧同名，原来的
// favorite_brand: 让同一个品牌在两侧永远是两个不同的字符串。
func (b *FashionUserBuilder) WithFavoriteBrands(brands ...string) *FashionUserBuilder {
	for _, brand := range brands {
		b.user.Labels = append(b.user.Labels, "brand:"+brand)
	}
	return b
}

func (b *FashionUserBuilder) WithComment(comment string) *FashionUserBuilder {
	b.user.Comment = comment
	return b
}

func (b *FashionUserBuilder) Build() User {
	return b.user
}

// FashionItemBuilder 时尚商品构建器
type FashionItemBuilder struct {
	item Item
}

func NewFashionItem(itemId string) *FashionItemBuilder {
	return &FashionItemBuilder{
		item: Item{
			ItemId:     itemId,
			IsHidden:   false,
			Categories: make([]string, 0),
			Labels:     make([]string, 0),
			Timestamp:  time.Now(),
		},
	}
}

func (b *FashionItemBuilder) WithCategories(categories ...string) *FashionItemBuilder {
	b.item.Categories = append(b.item.Categories, categories...)
	return b
}

func (b *FashionItemBuilder) WithBrand(brand string) *FashionItemBuilder {
	b.item.Labels = append(b.item.Labels, "brand:"+brand)
	return b
}

func (b *FashionItemBuilder) WithSeason(season string) *FashionItemBuilder {
	b.item.Labels = append(b.item.Labels, "season:"+season)
	return b
}

func (b *FashionItemBuilder) WithStyle(styles ...string) *FashionItemBuilder {
	for _, style := range styles {
		b.item.Labels = append(b.item.Labels, "style:"+style)
	}
	return b
}

func (b *FashionItemBuilder) WithColor(colors ...string) *FashionItemBuilder {
	for _, color := range colors {
		b.item.Labels = append(b.item.Labels, "color:"+color)
	}
	return b
}

func (b *FashionItemBuilder) WithMaterial(material string) *FashionItemBuilder {
	b.item.Labels = append(b.item.Labels, "material:"+material)
	return b
}

func (b *FashionItemBuilder) WithPriceRange(priceRange string) *FashionItemBuilder {
	b.item.Labels = append(b.item.Labels, "price_range:"+priceRange)
	return b
}

func (b *FashionItemBuilder) WithOccasion(occasions ...string) *FashionItemBuilder {
	for _, occasion := range occasions {
		b.item.Labels = append(b.item.Labels, "occasion:"+occasion)
	}
	return b
}

func (b *FashionItemBuilder) WithComment(comment string) *FashionItemBuilder {
	b.item.Comment = comment
	return b
}

func (b *FashionItemBuilder) WithTimestamp(timestamp time.Time) *FashionItemBuilder {
	b.item.Timestamp = timestamp
	return b
}

func (b *FashionItemBuilder) Hidden() *FashionItemBuilder {
	b.item.IsHidden = true
	return b
}

func (b *FashionItemBuilder) Build() Item {
	return b.item
}
