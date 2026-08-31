package traits

import (
	"fmt"
	"os"
	"testing"

	"fashion-recommend/client"
	"fashion-recommend/database"
	"fashion-recommend/models"

	"github.com/stretchr/testify/require"
)

// 这个文件是 A6 的实测：证明「分数不进字符串」之后，Gorse 真的会把 user label
// 收进索引（NumUserLabels > 0），而不是只证明我们发出的字符串好看。
//
// 需要一个跑着的 Gorse，所以默认跳过：
//
//	GORSE_LIVE_TEST=1 go test ./traits/ -run TestLive -v
//
// 设计上是**同一次 master 重载里的对照实验**，不是先后跑两遍：
//
//   - fx_old_a / fx_old_b 带旧格式的 label（style:minimalist:0.8 / :0.4），
//     两人的偏好相同但分数不同，于是字符串不同 → 各自都是单例
//   - fx_new_a / fx_new_b 的 label 由修好的 convertTraitsToLabels 真实产出，
//     同样是偏好相同、分数不同 → 字符串相同 → 跨用户第二次出现
//
// 两组用户除了 label 形态以外完全一样，所以 NumUserLabels 的增量只能来自
// 新格式那一组。先后跑两遍做不到这一点：master 的任务周期、数据集快照都会变。
//
// 这四个用户没有任何 feedback，不会进入协同过滤，也不参与 eval 队列；
// 跑完用 TestLiveCleanupFixtures 删掉。

// fixtureTraits 两个用户共享 minimalist / white，但分数不同 —— 这正是旧实现
// 会把它们拆成不同字符串的地方。
var fixtureTraits = map[string]*database.TraitsData{
	"a": {
		StylePreferences: map[string]float64{"minimalist": 0.8},
		ColorPreferences: map[string]float64{"white": 0.55},
		PriceSensitivity: "low",
	},
	"b": {
		StylePreferences: map[string]float64{"minimalist": 0.4},
		ColorPreferences: map[string]float64{"white": 0.91},
		PriceSensitivity: "low",
	},
}

// oldFormatLabels 复刻修复前 convertTraitsToLabels 在**带分数的前缀**上的产出。
//
// 故意不包含 price:，尽管旧实现也发它。理由是 price 不带分数，旧实现在它上面
// 只是前缀名不同（price: vs price_range:），两个用户依然共享同一个字符串、
// 依然会被 Gorse 收进索引。把它放进对照组会让对照组也贡献一个 label，
// NumUserLabels 的增量就归因不到 label 形态上了。改名那部分由单元测试
// TestPriceRangeNamespace 覆盖，不需要活体实验。
func oldFormatLabels(t *database.TraitsData) []string {
	labels := []string{}
	for style, score := range t.StylePreferences {
		if score > 0.5 {
			labels = append(labels, fmt.Sprintf("style:%s:%.1f", style, score))
		}
	}
	for color, score := range t.ColorPreferences {
		if score > 0.5 {
			labels = append(labels, fmt.Sprintf("color:%s:%.1f", color, score))
		}
	}
	return labels
}

func liveGorse(t *testing.T) *client.GorseClient {
	t.Helper()
	if os.Getenv("GORSE_LIVE_TEST") != "1" {
		t.Skip("需要跑着的 Gorse，设置 GORSE_LIVE_TEST=1 启用")
	}
	endpoint := os.Getenv("GORSE_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:8088"
	}
	return client.NewGorseClient(endpoint, os.Getenv("GORSE_API_KEY"))
}

func TestLivePushFixtures(t *testing.T) {
	gorse := liveGorse(t)
	g := &GorseSync{minScore: defaultMinScore, maxPerPrefix: defaultMaxPerPrefix}

	var users []models.User
	for suffix, traits := range fixtureTraits {
		old := oldFormatLabels(traits)
		fixed := g.convertTraitsToLabels(traits)

		t.Logf("fx_old_%s labels = %v", suffix, old)
		t.Logf("fx_new_%s labels = %v", suffix, fixed)

		users = append(users,
			models.User{UserId: "fx_old_" + suffix, Labels: old,
				Comment: "day2 对照组：修复前的 label 格式"},
			models.User{UserId: "fx_new_" + suffix, Labels: fixed,
				Comment: "day2 实验组：修复后的 label 格式"},
		)
	}

	require.NoError(t, gorse.InsertUsers(users))

	// 采样读回，不看 dashboard 计数器 —— 那个是滞后的
	for _, u := range users {
		got, err := gorse.GetUser(u.UserId)
		require.NoError(t, err)
		require.ElementsMatch(t, u.Labels, got.Labels, "user %s", u.UserId)
	}

	// 对照组的两个用户之间不能有任何共享 label，实验组必须有 —— 这是
	// NumUserLabels 增量能归因到 label 形态的前提。
	require.Empty(t, intersect(oldFormatLabels(fixtureTraits["a"]), oldFormatLabels(fixtureTraits["b"])),
		"对照组不该有共享 label，否则实验就不成立了")
	require.ElementsMatch(t,
		[]string{"style:minimalist", "color:white", "price_range:budget"},
		intersect(g.convertTraitsToLabels(fixtureTraits["a"]), g.convertTraitsToLabels(fixtureTraits["b"])))

	t.Log("已推送。重启 master 触发数据集重载，再读 /api/dashboard/status 的 NumUserLabels")
}

func TestLiveCleanupFixtures(t *testing.T) {
	gorse := liveGorse(t)
	if os.Getenv("GORSE_CLEANUP") != "1" {
		t.Skip("设置 GORSE_CLEANUP=1 才会删除 fixture 用户")
	}
	for _, suffix := range []string{"a", "b"} {
		for _, prefix := range []string{"fx_old_", "fx_new_"} {
			// 用空 label 覆盖即可让它们从 label 索引里消失，不必删用户
			require.NoError(t, gorse.InsertUser(models.User{
				UserId: prefix + suffix, Labels: []string{}, Comment: "day2 fixture 已清理",
			}))
		}
	}
}

func intersect(a, b []string) []string {
	set := make(map[string]bool, len(a))
	for _, s := range a {
		set[s] = true
	}
	var out []string
	for _, s := range b {
		if set[s] {
			out = append(out, s)
		}
	}
	return out
}
